import weakref
from fparser.two import Fortran2003
from fparser.two.utils import get_child, walk

from loki.frontend import Frontend
from loki.frontend.omni import parse_omni_ast, parse_omni_source
from loki.frontend.ofp import parse_ofp_ast, parse_ofp_source
from loki.frontend.fparser import parse_fparser_ast, parse_fparser_source
from loki.ir import (
    Declaration, Allocation, Import, Section, CallStatement,
    CallContext, Intrinsic, Interface
)
from loki.expression import FindVariables, Array, Scalar, SubstituteExpressions
from loki.visitors import FindNodes, Transformer
from loki.tools import as_tuple, flatten
from loki.types import TypeTable


__all__ = ['Subroutine']


class Subroutine:
    """
    Class to handle and manipulate a single subroutine.

    :param name: Name of the subroutine
    :param ast: Frontend node for this subroutine
    :param raw_source: Raw source string, broken into lines(!), as it
                       appeared in the parsed source file.
    :param typedefs: Optional list of external definitions for derived
                     types that allows more detaild type information.
    :param symbols: Instance of class:``TypeTable`` used to cache type information
                    for all symbols defined within this module's scope.
    :param types: Instance of class:``TypeTable`` used to cache type information
                  for all (derived) types defined within this module's scope.
    :param parent: Optional enclosing scope, to which a weakref can be held for symbol lookup.
    """

    def __init__(self, name, args=None, docstring=None, spec=None, body=None,
                 members=None, ast=None, typedefs=None, bind=None, is_function=False,
                 symbols=None, types=None, parent=None):
        self.name = name
        self._ast = ast
        self._dummies = as_tuple(a.lower() for a in as_tuple(args))  # Order of dummy arguments

        self._parent = weakref.ref(parent) if parent is not None else None

        self.symbols = symbols
        if self.symbols is None:
            parent_symbols = self.parent.symbols if self.parent is not None else None
            self.symbols = TypeTable(parent=parent_symbols)

        self.types = types
        if self.types is None:
            parent_types = self.parent.types if self.parent is not None else None
            self.types = TypeTable(parent=parent_types)

        # The primary IR components
        self.docstring = docstring
        self.spec = spec
        self.body = body
        self._members = members

        self.bind = bind
        self.is_function = is_function

    @staticmethod
    def _infer_allocatable_shapes(spec, body):
        """
        Infer variable symbol shapes from allocations of ``allocatable`` arrays.
        """
        alloc_map = {}
        for alloc in FindNodes(Allocation).visit(body):
            for v in alloc.variables:
                if isinstance(v, Array):
                    if alloc.data_source:
                        alloc_map[v.name.lower()] = alloc.data_source.type.shape
                    else:
                        alloc_map[v.name.lower()] = v.dimensions.index_tuple
        vmap = {}
        for v in FindVariables().visit(body):
            if v.name.lower() in alloc_map:
                vtype = v.type.clone(shape=alloc_map[v.name.lower()])
                vmap[v] = v.clone(type=vtype)
        smap = {}
        for v in FindVariables().visit(spec):
            if v.name.lower() in alloc_map:
                vtype = v.type.clone(shape=alloc_map[v.name.lower()])
                smap[v] = v.clone(type=vtype)
        return SubstituteExpressions(smap).visit(spec), SubstituteExpressions(vmap).visit(body)

    @classmethod
    def from_source(cls, source, typedefs=None, xmods=None, frontend=Frontend.FP):
        """
        Create ``Subroutine`` entry node from raw source string using given frontend.

        :param source: Fortran source string
        :param typdedefs: Derived-type definitions from external modules
        :param xmods: Locations of "xmods" module directory for OMNI frontend
        :param frontend: Choice of frontend to use for parsing source (default FP)
        """
        # TODO: Enable pre-processing on-the-fly

        if frontend == Frontend.OMNI:
            ast = parse_omni_source(source, xmods=xmods)
            typetable = ast.find('typeTable')
            f_ast = ast.find('globalDeclarations/FfunctionDefinition')
            return cls.from_omni(ast=f_ast, raw_source=source,
                                 typetable=typetable, typedefs=typedefs)

        if frontend == Frontend.OFP:
            ast = parse_ofp_source(source)
            f_ast = ast.find('file/subroutine')
            return cls.from_ofp(ast=f_ast, raw_source=source, typedefs=typedefs)

        if frontend == Frontend.FP:
            ast = parse_fparser_source(source)
            f_ast = get_child(ast, Fortran2003.Subroutine_Subprogram)
            return cls.from_fparser(ast=f_ast, raw_source=source, typedefs=typedefs)

        raise NotImplementedError('Unknown frontend: %s' % frontend)

    @classmethod
    def from_ofp(cls, ast, raw_source, name=None, typedefs=None, pp_info=None, parent=None):
        name = name or ast.attrib['name']
        is_function = ast.tag == 'function'
        obj = cls(name=name, ast=ast, typedefs=typedefs, parent=parent, is_function=is_function)

        # Store the names of variables in the subroutine signature
        arg_ast = ast.findall('header/arguments/argument')
        args = [arg.attrib['name'].upper() for arg in arg_ast]

        # Decompose the body into known sections
        ast_body = list(ast.find('body'))
        ast_spec = ast.find('body/specification')
        idx_spec = ast_body.index(ast_spec)
        ast_docs = ast_body[:idx_spec]
        ast_body = ast_body[idx_spec+1:]

        # Create a IRs for the docstring and the declaration spec
        docs = parse_ofp_ast(ast_docs, pp_info=pp_info, raw_source=raw_source, scope=obj)
        spec = parse_ofp_ast(ast_spec, typedefs=typedefs, pp_info=pp_info, raw_source=raw_source,
                             scope=obj)

        # Generate the subroutine body with all shape and type info
        body = parse_ofp_ast(ast_body, pp_info=pp_info, raw_source=raw_source, scope=obj)

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        spec, body = cls._infer_allocatable_shapes(spec, body)

        # Parse "member" subroutines and functions recursively
        members = None
        if ast.find('members'):
            members = [Subroutine.from_ofp(ast=s, raw_source=raw_source, typedefs=typedefs,
                                           pp_info=pp_info, parent=obj)
                       for s in ast.findall('members/subroutine')]
            members += [Subroutine.from_ofp(ast=s, raw_source=raw_source, typedefs=typedefs,
                                            pp_info=pp_info, parent=obj)
                        for s in ast.findall('members/function')]

        obj.__init__(name=name, args=args, docstring=docs, spec=spec, body=body,
                     members=members, ast=ast, typedefs=typedefs, symbols=obj.symbols,
                     types=obj.types, parent=parent, is_function=is_function)
        return obj

    @classmethod
    def from_omni(cls, ast, raw_source, typetable, typedefs=None, name=None, symbol_map=None,
                  parent=None):
        name = name or ast.find('name').text
        # file = ast.attrib['file']
        type_map = {t.attrib['type']: t for t in typetable}
        symbol_map = symbol_map or {}
        symbol_map.update({s.attrib['type']: s for s in ast.find('symbols')})

        # Check if it is a function or a subroutine. There may be a better way to do
        # this but OMNI does not seem to make it obvious, thus checking the return type
        name_id = ast.find('name').attrib['type']
        is_function = name_id in type_map and type_map[name_id].attrib['return_type'] != 'Fvoid'

        obj = cls(name=name, parent=parent, is_function=is_function)

        # Get the names of dummy variables from the type_map
        fhash = ast.find('name').attrib['type']
        ftype = [t for t in typetable.findall('FfunctionType')
                 if t.attrib['type'] == fhash][0]
        args = as_tuple(name.text for name in ftype.findall('params/name'))

        # Generate spec
        spec = parse_omni_ast(ast.find('declarations'), typedefs=typedefs, type_map=type_map,
                              symbol_map=symbol_map, raw_source=raw_source, scope=obj)
        # TODO: this filtered out external declarations explicitly. Reason?
        # mapper = {d: None for d in FindNodes(Declaration).visit(spec)
        #           if d._source.file != file or next(iter(d.variables)) == name}

        # Filter out the declaration for the subroutine name but keep it for functions (since
        # this declares the return type)
        if not is_function:
            mapper = {d: None for d in FindNodes(Declaration).visit(spec)
                      if d.variables[0].name == name}
            spec = Section(body=Transformer(mapper).visit(spec))
        else:
            spec = Section(body=spec)

        # Insert the `implicit none` statement OMNI omits (slightly hacky!)
        f_imports = [im for im in FindNodes(Import).visit(spec) if not im.c_import]
        spec_body = list(spec.body)
        spec_body.insert(len(f_imports), Intrinsic(text='IMPLICIT NONE'))
        spec._update(body=as_tuple(spec_body))

        # Parse member functions properly
        contains = ast.find('body/FcontainsStatement')
        members = None
        if contains is not None:
            members = [Subroutine.from_omni(ast=s, typetable=typetable, typedefs=typedefs,
                                            symbol_map=symbol_map, raw_source=raw_source,
                                            parent=obj)
                       for s in contains.findall('FfunctionDefinition')]
            # Strip members from the XML before we proceed
            ast.find('body').remove(contains)

        # Convert the core kernel to IR
        body = as_tuple(parse_omni_ast(ast.find('body'), typedefs=typedefs,
                                       type_map=type_map, symbol_map=symbol_map,
                                       raw_source=raw_source, scope=obj))

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        spec, body = cls._infer_allocatable_shapes(spec, body)

        obj.__init__(name=name, args=args, docstring=None, spec=spec, body=body,
                     members=members, ast=ast, parent=parent, symbols=obj.symbols,
                     types=obj.types, is_function=is_function)
        return obj

    @classmethod
    def from_fparser(cls, ast, raw_source, name=None, typedefs=None, pp_info=None, parent=None):
        is_function = isinstance(ast, Fortran2003.Function_Subprogram)
        if is_function:
            routine_stmt = get_child(ast, Fortran2003.Function_Stmt)
            name = name or routine_stmt.items[1].tostr()
        else:
            routine_stmt = get_child(ast, Fortran2003.Subroutine_Stmt)
            name = name or routine_stmt.get_name().string

        obj = cls(name, parent=parent, is_function=is_function)

        dummy_arg_list = routine_stmt.items[2]
        args = [arg.string for arg in dummy_arg_list.items] if dummy_arg_list else []

        spec_ast = get_child(ast, Fortran2003.Specification_Part)
        if spec_ast:
            spec = parse_fparser_ast(spec_ast, pp_info=pp_info, typedefs=typedefs, scope=obj,
                                     raw_source=raw_source)
        else:
            spec = ()
        spec = Section(body=spec)

        body_ast = get_child(ast, Fortran2003.Execution_Part)
        if body_ast:
            body = as_tuple(parse_fparser_ast(body_ast, pp_info=pp_info, typedefs=typedefs,
                                              scope=obj, raw_source=raw_source))
        else:
            body = ()
        # body = Section(body=body)

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        spec, body = cls._infer_allocatable_shapes(spec, body)

        # Parse "member" subroutines recursively
        members = None
        contains_ast = get_child(ast, Fortran2003.Internal_Subprogram_Part)
        if contains_ast:
            routine_types = (Fortran2003.Subroutine_Subprogram, Fortran2003.Function_Subprogram)
            members = [Subroutine.from_fparser(ast=s, raw_source=raw_source, typedefs=typedefs,
                                               pp_info=pp_info, parent=obj)
                       for s in walk(contains_ast, routine_types)]

        obj.__init__(name=name, args=args, docstring=None, spec=spec, body=body, ast=ast,
                     members=members, symbols=obj.symbols, types=obj.types, parent=parent,
                     is_function=is_function)
        return obj

    @property
    def variables(self):
        """
        Return the variables (including arguments) declared in this subroutine
        """
        return as_tuple(flatten(decl.variables for decl in FindNodes(Declaration).visit(self.spec)))

    @variables.setter
    def variables(self, variables):
        """
        Set the variables property and ensure that the internal declarations match.

        Note that arguments also count as variables and therefore any
        removal from this list will also remove arguments from the subroutine signature.
        """
        # First map variables to existing declarations
        declarations = FindNodes(Declaration).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.variables)

        for v in as_tuple(variables):
            if v not in decl_map:
                # By default, append new variables to the end of the spec
                new_decl = Declaration(variables=[v])
                self.spec.append(new_decl)

        # Run through existing declarations and check that all variables still exist
        dmap = {}
        for decl in FindNodes(Declaration).visit(self.spec):
            new_vars = as_tuple(v for v in decl.variables if v in variables)
            if len(new_vars) > 0:
                decl._update(variables=new_vars)
            else:
                dmap[decl] = None  # Mark for removal
        # Remove all redundant declarations
        self.spec = Transformer(dmap).visit(self.spec)

        # Filter the dummy list in case we removed an argument
        varnames = [str(v.name).lower() for v in variables]
        self._dummies = as_tuple(arg for arg in self._dummies if str(arg).lower() in varnames)

    @property
    def arguments(self):
        """
        Return arguments in order of the defined signature (dummy list).
        """
        # TODO: Can be simplified once we can directly lookup variables objects in scope
        arg_map = {v.name.lower(): v for v in self.variables if v.name.lower() in self._dummies}
        return as_tuple(arg_map[a.lower()] for a in self._dummies)

    @arguments.setter
    def arguments(self, arguments):
        """
        Set the arguments property and ensure that internal declarations and signature match.

        Note that removing arguments from this property does not actually remove declarations.
        """
        # First map variables to existing declarations
        declarations = FindNodes(Declaration).visit(self.spec)
        decl_map = dict((v, decl) for decl in declarations for v in decl.variables)

        arguments = as_tuple(arguments)
        for arg in arguments:
            if arg not in decl_map:
                # By default, append new variables to the end of the spec
                assert arg.type.intent is not None
                new_decl = Declaration(variables=[arg])
                self.spec.append(new_decl)

        # Set new dummy list according to input
        self._dummies = as_tuple(arg.name.lower() for arg in arguments)

    def enrich_calls(self, routines):
        """
        Attach target :class:`Subroutine` object to :class:`CallStatement`
        objects in the IR tree.

        :param call_targets: :class:`Subroutine` objects for corresponding
                             :class:`CallStatement` nodes in the IR tree.
        :param active: Additional flag indicating whether this :call:`CallStatement`
                       represents an active/inactive edge in the
                       interprocedural callgraph.
        """
        routine_map = {r.name.upper(): r for r in as_tuple(routines)}

        for call in FindNodes(CallStatement).visit(self.body):
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
    def members(self):
        """
        Tuple of member function defined in this `Subroutine`.
        """
        return as_tuple(self._members)

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
        """
        Interface object that defines the `Subroutine` signature in header files.
        """
        arg_names = [arg.name for arg in self.arguments]

        # Remove all local variable declarations from interface routine spec
        decl_map = {decl: None
                    for decl in FindNodes(Declaration).visit(self.spec)
                    if not all(v.name in arg_names for v in decl.variables)}
        spec = Transformer(decl_map).visit(self.spec)

        # Create the "interface routine" with all but declarations stripped
        routine = Subroutine(name=self.name, args=arg_names, spec=spec, body=None)
        return Interface(body=(routine,))

    @property
    def parent(self):
        """
        Access the enclosing scope.
        """
        return self._parent() if self._parent is not None else None

    def apply(self, op, **kwargs):
        """
        Apply a given transformation to the source file object.

        Note that the dispatch routine `op.apply(source)` will ensure
        that all entities of this `SourceFile` are correctly traversed.
        """
        # TODO: Should type-check for an `Operation` object here
        op.apply(self, **kwargs)
