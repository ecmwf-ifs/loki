from collections import OrderedDict
from fastcache import clru_cache
from sympy.core.cache import SYMPY_CACHE_SIZE, cacheit

from loki.frontend.parse import parse, OFP, OMNI
from loki.frontend.preprocessing import blacklist
from loki.ir import (Declaration, Allocation, Import, Section, Call,
                     CallContext, CommentBlock, Intrinsic)
from loki.expression import Variable, FindVariables, Array, Scalar, _symbol_type
from loki.types import BaseType, DerivedType
from loki.visitors import FindNodes, Transformer
from loki.tools import as_tuple


__all__ = ['Subroutine']


class InterfaceBlock(object):

    def __init__(self, name, arguments, imports, declarations):
        self.name = name
        self.arguments = arguments
        self.imports = imports
        self.declarations = declarations


class Subroutine(object):
    """
    Class to handle and manipulate a single subroutine.

    :param name: Name of the subroutine
    :param ast: OFP parser node for this subroutine
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    :param typedefs: Optional list of external definitions for derived
                     types that allows more detaild type information.
    """

    def __init__(self, name, args=None, docstring=None, spec=None, body=None,
                 members=None, ast=None, typedefs=None, bind=None, is_function=False):
        self.name = name
        self._ast = ast
        self._dummies = as_tuple(a.lower() for a in as_tuple(args))  # Order of dummy arguments

        self.arguments = None
        self.variables = None
        self._decl_map = None

        self.docstring = docstring
        self.spec = spec
        self.body = body
        self.members = members

        # Internalize argument declarations
        self._internalize()

        # Enrich internal representation with meta-data
        self._attach_derived_types(typedefs=typedefs)
        # self._derive_variable_shape(typedefs=typedefs)

        self.bind = bind
        self.is_function = is_function

        # Instantiate local symbol caches
        self._symbol_type_cache = cacheit(_symbol_type)
        self._array_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                       unhashable='ignore')(Array.__new_stage2__)
        self._scalar_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                        unhashable='ignore')(Scalar.__new_stage2__)


    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, typedefs=None, pp_info=None):
        name = name or ast.attrib['name']

        # Store the names of variables in the subroutine signature
        arg_ast = ast.findall('header/arguments/argument')
        args = [arg.attrib['name'].upper() for arg in arg_ast]

        # Create a IRs for declarations section and the loop body
        body = parse(ast.find('body'), raw_source=raw_source, frontend=OFP)

        # Apply postprocessing rules to re-insert information lost during preprocessing
        for r_name, rule in blacklist.items():
            info = pp_info[r_name] if pp_info is not None and r_name in pp_info else None
            body = rule.postprocess(body, info)

        # Parse "member" subroutines recursively
        members = None
        if ast.find('members'):
            members = [Subroutine.from_ofp(ast=s, raw_source=raw_source,
                                           typedefs=typedefs, pp_info=pp_info)
                       for s in ast.findall('members/subroutine')]

        # Separate docstring and declarations
        docstring = body[0] if isinstance(body[0], CommentBlock) else None
        spec = FindNodes(Section).visit(body)[0]
        body = Transformer({docstring: None, spec: None}).visit(body)

        return cls(name=name, args=args, docstring=docstring, spec=spec,
                   body=body, members=members, ast=ast, typedefs=typedefs)

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, symbol_map=None, typedefs=None):
        name = name or ast.find('name').text
        file = ast.attrib['file']
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        # Get the names of dummy variables from the type_map
        fhash = ast.find('name').attrib['type']
        ftype = [t for t in typetable.findall('FfunctionType')
                 if t.attrib['type'] == fhash][0]
        args = as_tuple(name.text for name in ftype.findall('params/name'))

        # Generate spec, filter out external declarations and docstring
        spec = parse(ast.find('declarations'), type_map=type_map,
                     symbol_map=symbol_map, raw_source=raw_source, frontend=OMNI)
        mapper = {d: None for d in FindNodes(Declaration).visit(spec)
                  if d._source.file != file or d.variables[0] == name}
        spec = Section(body=Transformer(mapper).visit(spec))

        # Insert the `implicit none` statement OMNI omits (slightly hacky!)
        f_imports = [im for im in FindNodes(Import).visit(spec) if not im.c_import]
        spec_body = list(spec.body)
        spec_body.insert(len(f_imports), Intrinsic(text='IMPLICIT NONE'))
        spec._update(body=as_tuple(spec_body))

        # TODO: Parse member functions properly
        contains = ast.find('body/FcontainsStatement')
        members = None
        if contains is not None:
            members = [Subroutine.from_omni(ast=s, typetable=typetable, symbol_map=symbol_map,
                                            typedefs=typedefs, raw_source=raw_source)
                       for s in contains]
            # Strip members from the XML before we proceed
            ast.find('body').remove(contains)

        # Convert the core kernel to IR
        body = parse(ast.find('body'), type_map=type_map, symbol_map=symbol_map,
                     raw_source=raw_source, frontend=OMNI)

        return cls(name=name, args=args, docstring=None, spec=spec, body=body,
                   members=members, ast=ast, typedefs=typedefs)

    def Variable(self, *args, **kwargs):
        # Here, we emulate Var.__new__, but we call the 2nd stage through
        # the locally cached decorator. This means we need
        name = kwargs.pop('name')
        dimensions = kwargs.pop('dimensions', None)
        parent = kwargs.pop('parent', None)

        # Create a new object from the static constructor with local
        # caching on `Kernel` instance!

        if dimensions is None:
            cls = self._symbol_type_cache(Scalar, name, parent)
            newobj = self._scalar_cache(cls, name, parent=parent)
        else:
            cls = self._symbol_type_cache(Array, name, parent)
            newobj = self._array_cache(cls, name, dimensions, parent=parent)

        # Since we are not actually using the object instation
        # mechanism, we need to call __init__ ourselves.
        newobj.__init__(*args, **kwargs)
        return newobj

    def _internalize(self):
        """
        Internalize argument and variable declarations.
        """
        self.arguments = [None] * len(self._dummies)
        self.variables = []
        self._decl_map = OrderedDict()
        dmap = {}

        for decl in FindNodes(Declaration).visit(self.ir):
            # Propagate dimensions to variables
            dvars = as_tuple(decl.variables)
            if decl.dimensions is not None:
                for v in dvars:
                    v.dimensions = decl.dimensions

            # Record all variables independently
            self.variables += list(dvars)

            # Insert argument variable at the position of the dummy
            for v in dvars:
                if v.name.lower() in self._dummies:
                    idx = self._dummies.index(v.name.lower())
                    self.arguments[idx] = v

            # Stash declaration and mark for removal
            for v in dvars:
                self._decl_map[v] = decl
            dmap[decl] = None

        # Remove declarations from the IR
        self.spec = Transformer(dmap).visit(self.spec)

    def _externalize(self, c_backend=False):
        """
        Re-insert argument declarations...
        """
        # A hacky way to ensure we don;t do this twice
        # TODO; Need better way to determine this; THIS IS NOT SAFE!
        if self._decl_map is None:
            return

        decls = []
        for v in self.variables:
            if c_backend and v in self.arguments:
                continue

            if v in self._decl_map:
                d = self._decl_map[v].clone()
                d.variables = as_tuple(v)
            else:
                d = Declaration(variables=[v], type=v.type)

            # Dimension declarations are done on variables
            d.dimensions = None

            decls += [d]
        self.spec.append(decls)

        self._decl_map = None

    def enrich_calls(self, routines):
        """
        Attach target :class:`Subroutine` object to :class:`Call`
        objects in the IR tree.

        :param call_targets: :class:`Subroutine` objects for corresponding
                             :class:`Call` nodes in the IR tree.
        :param active: Additional flag indicating whether this :call:`Call`
                       represents an active/inactive edge in the
                       interprocedural callgraph.
        """
        routine_map = {r.name.upper(): r for r in as_tuple(routines)}

        for call in FindNodes(Call).visit(self.body):
            if call.name.upper() in routine_map:
                # Calls marked as 'reference' are inactive and thus skipped
                active = True
                if call.pragma is not None and call.pragma.keyword == 'loki':
                    active = not call.pragma.content.startswith('reference')

                context = CallContext(routine=routine_map[call.name.upper()],
                                      active=active)
                call._update(context=context)

        # TODO: Could extend this to module and header imports to
        # facilitate user-directed inlining.

    def _attach_derived_types(self, typedefs=None):
        """
        Attaches the derived type definition from external header
        files to all :class:`Variable` instances (in-place).
        """
        for v in self.variables:
            if typedefs is not None and v.type is not None and v.type.name.upper() in typedefs:
                typedef = typedefs[v.type.name.upper()]
                derived_type = DerivedType(name=typedef.name, variables=typedef.variables,
                                           intent=v.type.intent, allocatable=v.type.allocatable,
                                           pointer=v.type.pointer, optional=v.type.optional)
                v._type = derived_type

    def _derive_variable_shape(self, typedefs=None):
        """
        Propgates the allocated dimensions (shape) from variable
        declarations to :class:`Variables` instances in the code body.

        :param ir: The control-flow IR into which to inject shape info
        :param declarations: Optional list of :class:`Declaration`s from
                             which to get shape dimensions.
        :param typdefs: Optional, additional derived-type definitions
                        from which to infer sub-variable shapes.

        Note, the shape derivation from derived types is currently
        limited to first-level nesting only.
        """
        typedefs = typedefs or {}

        # Create map of variable names to allocated shape (dimensions)
        # Make sure you capture sub-variables.
        shapes = {}
        derived = {}
        for v in self.variables:
            if v.type.name.upper() in typedefs:
                derived[v.name.upper()] = typedefs[v.type.name.upper()]

            if v.dimensions is not None:
                if v.shape is None:
                    # First derivation of shape goes from allcoated dimensions
                    shapes[v.name] = v.dimensions if len(v.dimensions) > 0 else None
                else:
                    # If shape is already set (by this routine or otherwise)
                    # we do not override it but propagate to instances.
                    shapes[v.name] = v.shape

                # Note: The forward propagation of the shapes is a clear
                # design flaw, as we end up calling this routine in two
                # contexts:
                # a) First, propagate the declared dimensions to all
                #    variable instances in the routine.
                # b) After IPA infers argument shapes from caller propagate
                #    these to all variable instancesin the routine.
                #
                # TODO: This should be fixed with kernel-level caching
                # of symbols; so that the variable from the argument
                # declaration aliases with each symbolic instance of
                # the variable within a kernel.


        # Override shapes for deferred-shape allocations
        for alloc in FindNodes(Allocation).visit(self.body):
            for v in alloc.variables:
                shapes[v.name] = v.dimensions

        # Apply shapes to meta-data variables
        for v in self.variables:
            v._shape = shapes[v.name]

        # Apply shapes to all variables in the IR and all members (in-place)
        variable_instances = FindVariables(unique=False).visit(self.ir)
        for member in as_tuple(self.members):
            variable_instances += FindVariables(unique=False).visit(member.ir)
        for v in variable_instances:
            if v.name in shapes:
                v._shape = shapes[v.name]

            if v.ref is not None and v.ref.name.upper() in derived:
                # We currently only follow a single level of nesting
                typevars = {tv.name.upper(): tv for tv in derived[v.ref.name.upper()].variables}
                if v.name.upper() in typevars:
                    v._shape = typevars[v.name.upper()].shape

    @property
    def ir(self):
        """
        Intermediate representation (AST) of the body in this subroutine
        """
        return (self.docstring, self.spec, self.body)

    @property
    def argnames(self):
        return [a.name for a in self.arguments]

    @property
    def variable_map(self):
        """
        Map of variable names to `Variable` objects
        """
        return {v.name.lower(): v for v in self.variables}

    @property
    def interface(self):
        arguments = self.arguments
        declarations = tuple(d for d in FindNodes(Declaration).visit(self.spec)
                             if any(v in arguments for v in d.variables))

        # Collect unknown symbols that we might need to import
        undefined = set()
        anames = [a.name for a in arguments]
        for decl in declarations:
            # Add potentially unkown TYPE and KIND symbols to 'undefined'
            if decl.type.name.upper() not in BaseType._base_types:
                undefined.add(decl.type.name)
            if decl.type.kind and not decl.type.kind.isdigit():
                undefined.add(decl.type.kind)
            # Add (pure) variable dimensions that might be defined elsewhere
            for v in decl.variables:
                undefined.update([str(d) for d in v.dimensions
                                  if isinstance(d, Variable) and d not in anames])

        # Create a sub-list of imports based on undefined symbols
        imports = []
        for use in FindNodes(Import).visit(self.spec):
            symbols = tuple(s for s in use.symbols if s in undefined)
            if not use.c_import and len(as_tuple(use.symbols)) > 0:
                # TODO: Check that only modules defining derived types
                # are included here.
                imports += [Import(module=use.module, symbols=symbols)]

        return InterfaceBlock(name=self.name, imports=imports,
                              arguments=arguments, declarations=declarations)
