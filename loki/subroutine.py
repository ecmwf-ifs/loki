from loki.frontend.parse import parse
from loki.frontend.preprocessing import blacklist
from loki.frontend.omni2ir import convert_omni2ir
from loki.ir import (Declaration, Allocation, Import, TypeDef, Section,
                     Call, CallContext, CommentBlock, Intrinsic)
from loki.expression import Variable, ExpressionVisitor
from loki.types import BaseType, DerivedType
from loki.visitors import FindNodes, Visitor, Transformer
from loki.tools import flatten, as_tuple


__all__ = ['Subroutine', 'Module']


class InterfaceBlock(object):

    def __init__(self, name, arguments, imports, declarations):
        self.name = name
        self.arguments = arguments
        self.imports = imports
        self.declarations = declarations


class Module(object):
    """
    Class to handle and manipulate source modules.

    :param name: Name of the module
    :param ast: OFP parser node for this module
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    """

    def __init__(self, name=None, spec=None, routines=None,
                 ast=None, raw_source=None):
        self.name = name or ast.attrib['name']
        self.spec = spec
        self.routines = routines

        self._ast = ast
        self._raw_source = raw_source

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None):
        # Process module-level type specifications
        name = name or ast.attrib['name']

        # Parse type definitions into IR and store
        spec_ast = ast.find('body/specification')
        spec = parse(spec_ast, raw_source)

        # TODO: Add routine parsing
        routine_asts = ast.findall('members/subroutine')
        routines = tuple(Subroutine.from_ofp(ast, raw_source)
                         for ast in routine_asts)

        # Process pragmas to override deferred dimensions
        cls._process_pragmas(spec)

        return cls(name=name, spec=spec, routines=routines,
                   ast=ast, raw_source=raw_source)

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, symbol_map=None):
        name = name or ast.attrib['name']

        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        # Generate spec, filter out external declarations and insert `implicit none`
        spec = convert_omni2ir(ast.find('declarations'), type_map=type_map,
                               symbol_map=symbol_map, raw_source=raw_source)
        spec = Section(body=spec)

        # TODO: Parse member functions properly
        contains = ast.find('FcontainsStatement')
        routines = None
        if contains is not None:
            routines = [Subroutine.from_omni(ast=s, typetable=typetable,
                                             symbol_map=symbol_map,
                                             raw_source=raw_source)
                        for s in contains]

        return cls(name=name, spec=spec, routines=routines, ast=ast)

    @classmethod
    def _process_pragmas(self, spec):
        """
        Process any '!$loki dimension' pragmas to override deferred dimensions
        """
        for typedef in FindNodes(TypeDef).visit(spec):
            pragmas = {p._source.lines[0]: p for p in typedef.pragmas}
            for decl in typedef.declarations:
                # Map pragmas by declaration line, not var line
                if decl._source.lines[0]-1 in pragmas:
                    pragma = pragmas[decl._source.lines[0]-1]
                    for v in decl.variables:
                        if pragma.keyword == 'loki' and pragma.content.startswith('dimension'):
                            # Found dimension override for variable
                            dims = pragma._source.string.split('dimension(')[-1]
                            dims = dims.split(')')[0].split(',')
                            dims = [d.strip() for d in dims]
                            # Override dimensions (hacky: not transformer-safe!)
                            v.dimensions = dims

    @property
    def typedefs(self):
        """
        Map of names and :class:`DerivedType`s defined in this module.
        """
        types = FindNodes(TypeDef).visit(self.spec)
        return {td.name.upper(): td for td in types}

    @property
    def subroutines(self):
        """
        List of :class:`Subroutine` objects that are members of this :class:`Module`.
        """
        return self.routines


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

    def __init__(self, name, args=None, docstring=None, spec=None,
                 body=None, members=None, ast=None):
        self.name = name

        self._argnames = args
        self._ast = ast

        self.docstring = docstring
        self.spec = spec
        self.body = body
        self.members = members

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, typedefs=None, pp_info=None):
        name = name or ast.attrib['name']

        # Store the names of variables in the subroutine signature
        arg_ast = ast.findall('header/arguments/argument')
        args = [arg.attrib['name'].upper() for arg in arg_ast]

        # Create a IRs for declarations section and the loop body
        body = parse(ast.find('body'), raw_source)

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

        obj = cls(name=name, args=args, docstring=docstring,
                  spec=spec, body=body, members=members, ast=ast)

        # Enrich internal representation with meta-data
        obj._attach_derived_types(typedefs=typedefs)
        obj._derive_variable_shape(typedefs=typedefs)

        return obj

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, name=None, symbol_map=None):
        name = name or ast.find('name').text
        file = ast.attrib['file']
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        # Get the names of dummy variables from the type_map
        fhash = ast.find('name').attrib['type']
        ftype = [t for t in typetable.findall('FfunctionType')
                 if t.attrib['type'] == fhash][0]
        args = as_tuple(name.text for name in ftype.findall('params/name'))

        # Generate spec, filter out external declarations
        spec = convert_omni2ir(ast.find('declarations'), type_map=type_map,
                               symbol_map=symbol_map, raw_source=raw_source)
        mapper = {d: None for d in FindNodes(Declaration).visit(spec)
                  if d._source['file'] != file or d.variables[0] == name}
        spec = Section(body=Transformer(mapper).visit(spec))

        # TODO: Parse member functions properly
        contains = ast.find('body/FcontainsStatement')
        members = None
        if contains is not None:
            members = [Subroutine.from_omni(ast=s, typetable=typetable,
                                            symbol_map=symbol_map,
                                            raw_source=raw_source)
                       for s in contains]
            # Strip members from the XML before we proceed
            ast.find('body').remove(contains)

        # Convert the core kernel to IR
        body = convert_omni2ir(ast.find('body'), type_map=type_map,
                               symbol_map=symbol_map, raw_source=raw_source)

        obj = cls(name=name, args=args, docstring=None, spec=spec, body=body,
                  members=members, ast=ast)

        return obj

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
            if typedefs is not None and v.type is not None and v.type.name in typedefs:
                typedef = typedefs[v.type.name]
                derived_type = DerivedType(name=typedef.name, variables=typedef.variables,
                                           intent=v.type.intent, allocatable=v.type.allocatable,
                                           pointer=v.type.pointer, optional=v.type.optional)
                v._type = derived_type

    def _derive_variable_shape(self, declarations=None, typedefs=None):
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
        declarations = declarations or FindNodes(Declaration).visit(self.spec)
        typedefs = typedefs or {}

        # Create map of variable names to allocated shape (dimensions)
        # Make sure you capture sub-variables.
        shapes = {}
        derived = {}
        for decl in declarations:
            if decl.type.name in typedefs:
                derived.update({v.name: typedefs[decl.type.name]
                                for v in decl.variables})

            if decl.dimensions is not None:
                shapes.update({v.name: decl.dimensions for v in decl.variables})
            else:
                shapes.update({v.name: v.dimensions for v in decl.variables
                               if v.dimensions is not None and len(v.dimensions) > 0})

        # Override shapes for deferred-shape allocations
        for alloc in FindNodes(Allocation).visit(self.body):
            shapes[alloc.variable.name] = alloc.variable.dimensions

        class VariableShapeInjector(ExpressionVisitor, Visitor):
            """
            Attach shape information to :class:`Variable` via the
            ``.shape`` attribute.
            """
            def __init__(self, shapes, derived):
                super(VariableShapeInjector, self).__init__()
                self.shapes = shapes
                self.derived = derived

            def visit_Variable(self, o):
                if o.name in self.shapes:
                    o._shape = self.shapes[o.name]

                if o.subvar is not None and o.name in self.derived:
                    # We currently only follow a single level of nesting
                    typevars = {v.name.upper(): v for v in self.derived[o.name].variables}
                    o.subvar._shape = typevars[o.subvar.name.upper()].dimensions

                # Recurse over children
                for c in o.children:
                    self.visit(c)

            def visit_Declaration(self, o):
                # Attach shape info to declaration dummy variables
                if o.type.allocatable:
                    for v in o.variables:
                        v._shape = self.shapes[v.name]

                # Recurse over children
                for c in o.children:
                    self.visit(c)

        # Apply dimensions via expression visitor (in-place)
        ir = (self.spec, self.body)
        VariableShapeInjector(shapes=shapes, derived=derived).visit(ir)

    @property
    def ir(self):
        """
        Intermediate representation (AST) of the body in this subroutine
        """
        return (self.docstring, self.spec, self.body)

    @property
    def argnames(self):
        return self._argnames

    @property
    def arguments(self):
        """
        List of argument names as defined in the subroutine signature.
        """
        vmap = self.variable_map
        return [vmap[name.upper()] for name in self.argnames]

    @property
    def variables(self):
        """
        List of all declared variables
        """
        decls = FindNodes(Declaration).visit(self.spec)
        return flatten([d.variables for d in decls])

    @property
    def variable_map(self):
        """
        Map of variable names to `Variable` objects
        """
        return {v.name.upper(): v for v in self.variables}

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
        for use in self.imports:
            symbols = tuple(s for s in use.symbols if s in undefined)
            if len(symbols) > 0:
                imports += [Import(module=use.module, symbols=symbols)]

        return InterfaceBlock(name=self.name, imports=imports,
                              arguments=arguments, declarations=declarations)
