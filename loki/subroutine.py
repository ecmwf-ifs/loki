from collections import OrderedDict
from fparser.two import Fortran2003
from fparser.two.utils import get_child

from loki.frontend.omni import parse_omni_ast
from loki.frontend.ofp import parse_ofp_ast
from loki.frontend.fparser import parse_fparser_ast
from loki.ir import (Declaration, Allocation, Import, Section, Call,
                     CallContext, Intrinsic)
from loki.expression import FindVariables, Array, Scalar, SubstituteExpressions
from loki.types import BaseType
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
                 members=None, ast=None, cache=None, typedefs=None, bind=None,
                 is_function=False):
        self.name = name
        self._ast = ast
        self._dummies = as_tuple(a.lower() for a in as_tuple(args))  # Order of dummy arguments

        # Symbol caching by default happens per subroutine
        self._cache = None  # cache or SymbolCache()

        self.arguments = None
        self.variables = None
        self._decl_map = None

        self.docstring = docstring
        self.spec = spec
        self.body = body
        self.members = members

        # Internalize argument declarations
        self._internalize()

        self.bind = bind
        self.is_function = is_function

    @classmethod
    def _infer_allocatable_shapes(cls, spec, body):
        """
        Infer variable symbol shapes from allocations of ``allocatable`` arrays.
        """
        alloc_map = {}
        for alloc in FindNodes(Allocation).visit(body):
            for v in alloc.variables:
                if isinstance(v, Array):
                    alloc_map[v.name.lower()] = v.dimensions
        vmap = {}
        for v in FindVariables().visit(body):
            if v.name.lower() in alloc_map:
                vmap[v] = v.clone(shape=alloc_map[v.name.lower()])
        smap = {}
        for v in FindVariables().visit(spec):
            if v.name.lower() in alloc_map:
                smap[v] = v.clone(shape=alloc_map[v.name.lower()])
        return SubstituteExpressions(smap).visit(spec), SubstituteExpressions(vmap).visit(body)

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, typedefs=None, pp_info=None, cache=None):
        name = name or ast.attrib['name']

        # Store the names of variables in the subroutine signature
        arg_ast = ast.findall('header/arguments/argument')
        args = [arg.attrib['name'].upper() for arg in arg_ast]

        cache = None  # SymbolCache() if cache is None else cache

        # Decompose the body into known sections
        ast_body = list(ast.find('body'))
        ast_spec = ast.find('body/specification')
        idx_spec = ast_body.index(ast_spec)
        ast_docs = ast_body[:idx_spec]
        ast_body = ast_body[idx_spec+1:]

        # Create a IRs for the docstring and the declaration spec
        docs = parse_ofp_ast(ast_docs, cache=cache, pp_info=pp_info, raw_source=raw_source)
        spec = parse_ofp_ast(ast_spec, cache=cache, typedefs=typedefs,
                             pp_info=pp_info, raw_source=raw_source)

        # Derive type and shape maps to propagate through the subroutine body
        type_map = {}
        shape_map = {}
        for decl in FindNodes(Declaration).visit(spec):
            type_map.update({v.name: v.type for v in decl.variables})
            shape_map.update({v.name: v.shape for v in decl.variables
                              if isinstance(v, Array)})

        # Generate the subroutine body with all shape and type info
        body = parse_ofp_ast(ast_body, cache=cache, pp_info=pp_info,
                             shape_map=shape_map, type_map=type_map,
                             raw_source=raw_source)

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        spec, body = cls._infer_allocatable_shapes(spec, body)

        # Parse "member" subroutines recursively
        members = None
        if ast.find('members'):
            members = [Subroutine.from_ofp(ast=s, raw_source=raw_source, cache=cache,
                                           typedefs=typedefs, pp_info=pp_info)
                       for s in ast.findall('members/subroutine')]

        return cls(name=name, args=args, docstring=docs, spec=spec, body=body,
                   members=members, ast=ast, typedefs=typedefs, cache=cache)

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, typedefs=None, name=None, symbol_map=None):
        name = name or ast.find('name').text
        file = ast.attrib['file']
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {s.attrib['type']: s for s in ast.find('symbols')}

        # Get the names of dummy variables from the type_map
        fhash = ast.find('name').attrib['type']
        ftype = [t for t in typetable.findall('FfunctionType')
                 if t.attrib['type'] == fhash][0]
        args = as_tuple(name.text for name in ftype.findall('params/name'))

        cache = None  # SymbolCache()

        # Generate spec, filter out external declarations and docstring
        spec = parse_omni_ast(ast.find('declarations'), typedefs=typedefs, type_map=type_map,
                              symbol_map=symbol_map, cache=cache, raw_source=raw_source)
        mapper = {d: None for d in FindNodes(Declaration).visit(spec)
                  if d._source.file != file or d.variables[0].name == name}
        spec = Section(body=Transformer(mapper).visit(spec))

        # Insert the `implicit none` statement OMNI omits (slightly hacky!)
        f_imports = [im for im in FindNodes(Import).visit(spec) if not im.c_import]
        spec_body = list(spec.body)
        spec_body.insert(len(f_imports), Intrinsic(text='IMPLICIT NONE'))
        spec._update(body=as_tuple(spec_body))

        # Get the declared shapes of local variables and arguments
        shape_map = {}
        for decl in FindNodes(Declaration).visit(spec):
            for v in decl.variables:
                if isinstance(v, Array):
                    shape_map[v.name] = v.shape

        # Parse member functions properly
        contains = ast.find('body/FcontainsStatement')
        members = None
        if contains is not None:
            members = [Subroutine.from_omni(ast=s, typetable=typetable, typedefs=typedefs,
                                            symbol_map=symbol_map, raw_source=raw_source)
                       for s in contains]
            # Strip members from the XML before we proceed
            ast.find('body').remove(contains)

        # Convert the core kernel to IR
        body = as_tuple(parse_omni_ast(ast.find('body'), cache=cache, typedefs=typedefs,
                                       type_map=type_map, symbol_map=symbol_map,
                                       shape_map=shape_map, raw_source=raw_source))

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        spec, body = cls._infer_allocatable_shapes(spec, body)

        return cls(name=name, args=args, docstring=None, spec=spec, body=body,
                   members=members, ast=ast, cache=cache)

    @classmethod
    def from_fparser(cls, ast, name=None, typedefs=None):
        routine_stmt = get_child(ast, Fortran2003.Subroutine_Stmt)
        name = name or routine_stmt.get_name().string
        dummy_arg_list = routine_stmt.items[2]
        args = [arg.string for arg in dummy_arg_list.items]

        cache = None  # SymbolCache()

        spec_ast = get_child(ast, Fortran2003.Specification_Part)
        spec = parse_fparser_ast(spec_ast, typedefs=typedefs, cache=cache)
        spec = Section(body=spec)

        # Derive type and shape maps to propagate through the subroutine body
        type_map = {}
        shape_map = {}
        for decl in FindNodes(Declaration).visit(spec):
            type_map.update({v.name: v.type for v in decl.variables})
            shape_map.update({v.name: v.shape for v in decl.variables
                              if isinstance(v, Array)})

        body_ast = get_child(ast, Fortran2003.Execution_Part)
        body = parse_fparser_ast(body_ast, shape_map=shape_map, type_map=type_map, cache=cache)
        # body = Section(body=body)

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        spec, body = cls._infer_allocatable_shapes(spec, body)

        return cls(name=name, args=args, docstring=None, spec=spec, body=body, ast=ast, cache=cache)

#    def Variable(self, *args, **kwargs):
#        """
#        Instantiate cached variable symbols from local symbol cache.
#        """
#        return self._cache.Variable(*args, **kwargs)

    def _internalize(self):
        """
        Internalize argument and variable declarations.
        """
        self.arguments = [None] * len(self._dummies)
        self.variables = []
        self._decl_map = OrderedDict()
        dmap = {}

        for decl in FindNodes(Declaration).visit(self.ir):
            # Record all variables independently
            self.variables += list(decl.variables)

            # Insert argument variable at the position of the dummy
            for v in decl.variables:
                if v.name.lower() in self._dummies:
                    idx = self._dummies.index(v.name.lower())
                    self.arguments[idx] = v

            # Stash declaration and mark for removal
            for v in decl.variables:
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
                if isinstance(v, Array):
                    undefined.update([str(d) for d in v.dimensions
                                      if isinstance(d, Scalar) and d not in anames])

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
