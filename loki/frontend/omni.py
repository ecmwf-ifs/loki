from pathlib import Path
from shutil import which
import xml.etree.ElementTree as ET

from loki.frontend.source import Source
from loki.frontend.util import OMNI, sanitize_ir
from loki.visitors import GenericVisitor, FindNodes, Transformer
from loki import ir
from loki.expression import (
    symbols as sym, operations as op,
    ExpressionDimensionsMapper, StringConcat, AttachScopesMapper
)
from loki.logging import info, debug, DEBUG, warning, error
from loki.config import config
from loki.tools import (
    as_tuple, timeit, execute, gettempdir, filehash, CaseInsensitiveDict
)
from loki.pragma_utils import (
    process_dimension_pragmas, pragmas_attached
)
from loki.types import BasicType, DerivedType, ProcedureType, SymbolAttributes


__all__ = ['HAVE_OMNI', 'parse_omni_source', 'parse_omni_file', 'parse_omni_ast']


HAVE_OMNI = which('F_Front') is not None
"""Indicate whether OMNI frontend is available."""


@timeit(log_level=DEBUG)
def parse_omni_file(filename, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to generate the OMNI AST.

    Note that the intermediate XML files can be dumped to file via by setting
    the environment variable ``LOKI_OMNI_DUMP_XML``.
    """
    if not HAVE_OMNI:
        error('OMNI is not available. Is "F_Front" in the search path?')

    dump_xml_files = config['omni-dump-xml']

    filepath = Path(filename)
    info(f'[Loki::OMNI] Parsing {filepath}')

    xml_path = filepath.with_suffix('.xml')
    xmods = xmods or []

    cmd = ['F_Front', '-fleave-comment']
    for m in xmods:
        cmd += ['-M', f'{Path(m)}']
    cmd += [f'{filepath}']

    if dump_xml_files:
        # Parse AST from xml file dumped to disk
        cmd += ['-o', f'{xml_path}']
        execute(cmd)
        return ET.parse(str(xml_path)).getroot()

    result = execute(cmd, silent=False, capture_output=True, text=True)
    return ET.fromstring(result.stdout)


@timeit(log_level=DEBUG)
def parse_omni_source(source, filepath=None, xmods=None):
    """
    Deploy the OMNI compiler's frontend (F_Front) to AST for a source string.
    """
    # Use basename of filepath if given
    if filepath is None:
        filepath = Path(filehash(source, prefix='omni-', suffix='.f90'))
    else:
        filepath = filepath.with_suffix(f'.omni{filepath.suffix}')

    # Always store intermediate flies in tmp dir
    filepath = gettempdir()/filepath.name

    debug(f'[Loki::OMNI] Writing temporary source {filepath}')
    with filepath.open('w') as f:
        f.write(source)

    return parse_omni_file(filename=filepath, xmods=xmods)


@timeit(log_level=DEBUG)
def parse_omni_ast(ast, definitions=None, type_map=None, symbol_map=None,
                   raw_source=None, scope=None):
    """
    Generate an internal IR from the raw OMNI parser AST.
    """
    # Parse the raw OMNI language AST
    _ir = OMNI2IR(type_map=type_map, definitions=definitions, symbol_map=symbol_map,
                  raw_source=raw_source, scope=scope).visit(ast)

    # Perform some minor sanitation tasks
    _ir = sanitize_ir(_ir, OMNI)

    return _ir


class OMNI2IR(GenericVisitor):
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    _omni_types = {
        'Fint': 'INTEGER',
        'Freal': 'REAL',
        'Flogical': 'LOGICAL',
        'Fcharacter': 'CHARACTER',
        'Fcomplex': 'COMPLEX',
        'int': 'INTEGER',
        'real': 'REAL',
    }

    def __init__(self, definitions=None, type_map=None, symbol_map=None,
                 raw_source=None, scope=None):
        super().__init__()

        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.type_map = type_map or {}
        self.symbol_map = symbol_map or {}
        self.raw_source = raw_source.splitlines(keepends=True)
        self.default_scope = scope

    @staticmethod
    def warn_or_fail(msg):
        if config['frontend-strict-mode']:
            error(msg)
            raise NotImplementedError
        warning(msg)

    def type_from_type_attrib(self, type_attrib, **kwargs):
        """
        Helper routine to derive :any:`SymbolAttributes` for a given type name/hash/id
        """
        if type_attrib in self._omni_types:
            typename = self._omni_types[type_attrib]
            _type = SymbolAttributes(BasicType.from_fortran_type(typename))
        elif type_attrib in self.type_map:
            _type = self.visit(self.type_map[type_attrib], **kwargs)
            dims = self.type_map[type_attrib].findall('indexRange')
            if dims:
                dimensions = as_tuple(self.visit(d, **kwargs) for d in dims)
                _type = _type.clone(shape=dimensions)
        else:
            _type = SymbolAttributes(BasicType.from_fortran_type(type_attrib))
        return _type

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        return super().lookup_method(instance)

    def get_source(self, o):
        """Helper method that builds the source object for a node"""
        file = o.attrib.get('file', None)
        lineno = o.attrib.get('lineno', None)
        if lineno:
            lineno = int(lineno)
            lines = (lineno, lineno)
            string = self.raw_source[lineno-1]
        else:
            lines = (None, None)
            string = None
        return Source(lines=lines, string=string, file=file)

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        kwargs['source'] = self.get_source(o)
        kwargs.setdefault('scope', self.default_scope)
        kwargs.setdefault('symbol_map', self.symbol_map)
        return super().visit(o, **kwargs)

    def visit_Element(self, o, **kwargs):
        """
        Universal default for XML element types
        """
        warning('No specific handler for node type %s', o.__class__.name)
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_XcodeProgram(self, o, **kwargs):
        body = [self.visit(c, **kwargs) for c in o.find('globalDeclarations')]
        return ir.Section(body=as_tuple(body))

    def visit_FuseDecl(self, o, **kwargs):
        # No ONLY list
        nature = 'intrinsic' if o.attrib.get('intrinsic') == 'true' else None
        name = o.attrib['name']
        scope = kwargs['scope']

        # Rename list
        rename_list = dict(self.visit(s, **kwargs) for s in o.findall('rename'))

        module = self.definitions.get(name, None)
        if module is not None:
            # Import symbol attributes from module, if available
            for k, v in module.symbol_attrs.items():
                if k in rename_list:
                    local_name = rename_list[k].name
                    scope.symbol_attrs[local_name] = v.clone(imported=True, module=module, use_name=k)
                else:
                    # Need to explicitly reset use_name in case we are importing a symbol
                    # that stems from an import with a rename-list
                    scope.symbol_attrs[k] = v.clone(imported=True, module=module, use_name=None)
        elif rename_list:
            # Module not available but some information via rename-list
            scope.symbol_attrs.update({v.name: v.type.clone(imported=True, use_name=k) for k, v in rename_list.items()})
        rename_list = tuple(rename_list.items()) if rename_list else None
        return ir.Import(module=name, nature=nature, rename_list=rename_list, c_import=False, source=kwargs['source'])

    def visit_FuseOnlyDecl(self, o, **kwargs):
        # ONLY list given (import only selected symbols)
        nature = 'intrinsic' if o.attrib.get('intrinsic') == 'true' else None
        name = o.attrib['name']
        scope = kwargs['scope']
        symbols = tuple(self.visit(c, **kwargs) for c in o.findall('renamable'))
        if nature == 'intrinsic':
            module = None
        else:
            module = self.definitions.get(name, None)
        if module is None:
            # Initialize symbol attributes as DEFERRED
            for s in symbols:
                if isinstance(s, tuple):  # Renamed symbol
                    scope.symbol_attrs[s[1].name] = SymbolAttributes(BasicType.DEFERRED, imported=True, use_name=s[0])
                else:
                    scope.symbol_attrs[s.name] = SymbolAttributes(BasicType.DEFERRED, imported=True)
        else:
            # Import symbol attributes from module
            for s in symbols:
                if isinstance(s, tuple):  # Renamed symbol
                    scope.symbol_attrs[s[1].name] = module.symbol_attrs[s[0]].clone(
                        imported=True, module=module, use_name=s[0]
                    )
                else:
                    # Need to explicitly reset use_name in case we are importing a symbol
                    # that stems from an import with a rename-list
                    scope.symbol_attrs[s.name] = module.symbol_attrs[s.name].clone(
                        imported=True, module=module, use_name=None
                    )
        symbols = tuple(
            s[1].rescope(scope=scope) if isinstance(s, tuple) else s.rescope(scope=scope) for s in symbols
        )
        return ir.Import(module=name, symbols=symbols, nature=nature, c_import=False, source=kwargs['source'])

    def visit_renamable(self, o, **kwargs):
        name = o.attrib['use_name']
        if o.attrib.get('is_operator') == 'true':
            if name == '=':
                name = 'ASSIGNMENT(=)'
            else:
                name = f'OPERATOR({name})'

        if o.attrib.get('local_name'):
            return (name, sym.Variable(name=o.attrib['local_name'], source=kwargs['source']))
        return sym.Variable(name=name, source=kwargs['source'])

    visit_rename = visit_renamable

    def visit_FinterfaceDecl(self, o, **kwargs):
        abstract = o.get('is_abstract') == 'true'

        if o.get('is_assignment') == 'true':
            name = 'ASSIGNMENT(=)'
        elif o.get('is_operator') == 'true':
            name = f'OPERATOR({o.get("name")})'
        else:
            name = o.get('name')

        if name is not None:
            scope = kwargs['scope']
            if name not in scope.symbol_attrs:
                scope.symbol_attrs[name] = SymbolAttributes(ProcedureType(name, is_generic=True))
            spec = sym.Variable(name=name, scope=kwargs['scope'])
        else:
            spec = None

        body = tuple(self.visit(c, **kwargs) for c in o)
        return ir.Interface(body=body, abstract=abstract, spec=spec, source=kwargs['source'])

    def _create_Subroutine_object(self, o, scope, symbol_map):
        """Helper method to instantiate a Subroutine object"""
        from loki.subroutine import Subroutine  # pylint: disable=import-outside-toplevel
        assert o.tag in ('FfunctionDefinition', 'FfunctionDecl')
        name = o.find('name').text

        # Check if the Subroutine node has been created before by looking it up in the scope
        if scope is not None and name in scope.symbol_attrs:
            proc_type = scope.symbol_attrs[name]  # Look-up only in current scope!
            if proc_type and proc_type.dtype.procedure != BasicType.DEFERRED:
                return proc_type.dtype.procedure

        # Return type and dummy args
        ftype = self.type_map[o.find('name').attrib['type']]
        proc_type = self.visit(ftype, scope=scope, symbol_map=symbol_map)
        is_function = ftype.attrib['return_type'] != 'Fvoid'
        args = tuple(a.text for a in ftype.findall('params/name'))

        # Function/Subroutine prefix
        prefix = proc_type.prefix or ()
        if prefix:
            # We store the prefix on the Subroutine object, so let's remove it from the symbol attrs
            proc_type = proc_type.clone(prefix=None)

        # Instantiate the object
        routine = Subroutine(
            name=name, args=args, prefix=prefix, bind=None,
            is_function=is_function, parent=scope,
            ast=o, source=self.get_source(o)
        )

        return routine

    def visit_FfunctionDefinition(self, o, **kwargs):
        # Update the symbol map with local entries
        kwargs['symbol_map'] = kwargs['symbol_map'].copy()
        kwargs['symbol_map'].update({s.attrib['type']: s for s in o.find('symbols')})

        # Instantiate the object
        routine = self._create_Subroutine_object(o, kwargs['scope'], kwargs['symbol_map'])
        kwargs['scope'] = routine

        # Parse the spec
        spec = self.visit(o.find('declarations'), **kwargs)
        spec = sanitize_ir(spec, OMNI)

        # Filter out the declaration for the subroutine name but keep it for functions (since
        # this declares the return type)
        spec_map = {}
        if not routine.is_function:
            spec_map.update({
                d: None for d in FindNodes((ir.ProcedureDeclaration, ir.VariableDeclaration)).visit(spec)
                if routine.name in d.symbols
            })

        # Hack: We remove comments from the beginning of the spec to get the docstring
        docstring = []
        for node in spec.body:
            if node in spec_map:
                continue
            if not isinstance(node, (ir.Comment, ir.CommentBlock)):
                break
            docstring.append(node)
            spec_map[node] = None
        docstring = as_tuple(docstring)
        spec = Transformer(spec_map, invalidate_source=False).visit(spec)

        # Insert the `implicit none` statement OMNI omits (slightly hacky!)
        f_imports = [im for im in FindNodes(ir.Import).visit(spec) if not im.c_import]
        if not f_imports:
            spec.prepend(ir.Intrinsic(text='IMPLICIT NONE'))
        else:
            spec.insert(spec.body.index(f_imports[-1])+1, ir.Intrinsic(text='IMPLICIT NONE'))

        # Parse member functions
        body_ast = o.find('body')
        contains_ast = None if body_ast is None else body_ast.find('FcontainsStatement')
        if contains_ast is not None:
            contains = self.visit(contains_ast, **kwargs)

            # Strip contains part from the XML before we proceed
            body_ast.remove(contains_ast)
        else:
            contains = None

        # Finally, take care of the body
        if body_ast is None:
            body = ir.Section(body=())
        else:
            body = ir.Section(body=self.visit(body_ast, **kwargs))
            body = sanitize_ir(body, OMNI)

        # Finally, call the subroutine constructor on the object again to register all
        # bits and pieces in place and rescope all symbols
        # pylint: disable=unnecessary-dunder-call
        routine.__init__(
            name=routine.name, args=routine._dummies,
            docstring=docstring, spec=spec, body=body, contains=contains,
            ast=o, prefix=routine.prefix, bind=routine.bind, is_function=routine.is_function,
            rescope_symbols=True, parent=routine.parent, symbol_attrs=routine.symbol_attrs,
            source=routine.source, frontend=OMNI, incomplete=False
        )

        # Big, but necessary hack:
        # For deferred array dimensions on allocatables, we infer the conceptual
        # dimension by finding any `allocate(var(<dims>))` statements.
        routine.spec, routine.body = routine._infer_allocatable_shapes(routine.spec, routine.body)

        # Update array shapes with Loki dimension pragmas
        with pragmas_attached(routine, ir.VariableDeclaration):
            routine.spec = process_dimension_pragmas(routine.spec)

        return routine

    visit_FfunctionDecl = visit_FfunctionDefinition

    def visit_FcontainsStatement(self, o, **kwargs):
        body = [self.visit(c, **kwargs) for c in o]
        body = [c for c in body if c is not None]
        body = [ir.Intrinsic('CONTAINS', source=kwargs['source'])] + body
        return ir.Section(body=as_tuple(body))

    def visit_FmoduleProcedureDecl(self, o, **kwargs):
        symbols = as_tuple(self.visit(o.find('name'), **kwargs))
        symbols = AttachScopesMapper()(symbols, scope=kwargs['scope'])
        return ir.ProcedureDeclaration(symbols=symbols, module=True, source=kwargs.get('source'))

    def _create_Module_object(self, o, scope):
        """Helper method to instantiate a Module object"""
        from loki.module import Module  # pylint: disable=import-outside-toplevel

        name = o.attrib['name']

        # Check if the Module node has been created before by looking it up in the scope
        if scope is not None and name in scope.symbol_attrs:
            module_type = scope.symbol_attrs[name]  # Look-up only in current scope
            if module_type and module_type.dtype.module != BasicType.DEFERRED:
                return module_type.dtype.module

        return Module(name=name, parent=scope)


    def visit_FmoduleDefinition(self, o, **kwargs):
        # Update the symbol map with local entries
        kwargs['symbol_map'] = kwargs['symbol_map'].copy()
        kwargs['symbol_map'].update({s.attrib['type']: s for s in o.find('symbols')})

        # Instantiate the object
        module = self._create_Module_object(o, kwargs['scope'])
        kwargs['scope'] = module

        # Pre-populate symbol table with procedure types declared in this module
        # to correctly classify inline function calls and type-bound procedures
        contains_ast = o.find('FcontainsStatement')
        if contains_ast is not None:
            # Note that we overwrite this variable subsequently with the fully parsed subroutines
            # where the visit-method for the subroutine/function statement will pick out the existing
            # subroutine objects using the weakref pointers stored in the symbol table.
            # I know, it's not pretty but alternatively we could hand down this array as part of
            # kwargs but that feels like carrying around a lot of bulk, too.
            contains = [
                self._create_Subroutine_object(member_ast, kwargs['scope'], kwargs['symbol_map'])
                for member_ast in contains_ast.findall('FfunctionDefinition')
            ]

        # Parse the spec
        spec = self.visit(o.find('declarations'), **kwargs)
        spec = sanitize_ir(spec, OMNI)

        # Hack: We remove comments from the beginning of the spec to get the docstring
        docstring = []
        spec_map = {}
        for node in spec.body:
            if node in spec_map:
                continue
            if not isinstance(node, (ir.Comment, ir.CommentBlock)):
                break
            docstring.append(node)
            spec_map[node] = None
        docstring = as_tuple(docstring)
        spec = Transformer(spec_map, invalidate_source=False).visit(spec)

        # Parse member functions
        if contains_ast is not None:
            contains = self.visit(contains_ast, **kwargs)
        else:
            contains = None

        # Finally, call the module constructor on the object again to register all
        # bits and pieces in place and rescope all symbols
        # pylint: disable=unnecessary-dunder-call
        module.__init__(
            name=module.name, docstring=docstring, spec=spec, contains=contains,
            ast=o, rescope_symbols=True, source=kwargs['source'],
            parent=module.parent, symbol_attrs=module.symbol_attrs,
            frontend=OMNI, incomplete=False
        )

        return module

    def visit_declarations(self, o, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return ir.Section(body=body, source=kwargs['source'])

    def visit_body(self, o, **kwargs):
        body = tuple(self.visit(c, **kwargs) for c in o)
        body = tuple(c for c in body if c is not None)
        return body

    def visit_FimportDecl(self, o, **kwargs):
        symbols = tuple(self.visit(i, **kwargs) for i in o)
        symbols = AttachScopesMapper()(symbols, scope=kwargs['scope'])
        return ir.Import(
            module=None, symbols=symbols, f_import=True, source=kwargs['source']
        )

    def visit_varDecl(self, o, **kwargs):
        # OMNI has only one variable per declaration, find and create that
        name = o.find('name')
        variable = self.visit(name, **kwargs)

        # Create the declared type
        if name.attrib['type'] in self._omni_types:
            # Intrinsic scalar type
            t = self._omni_types[name.attrib['type']]
            _type = SymbolAttributes(BasicType.from_fortran_type(t))
            dimensions = None

        elif name.attrib['type'] in self.type_map:
            # Type with attributes or derived type
            tast = self.type_map[name.attrib['type']]
            _type = self.visit(tast, **kwargs)

            dimensions = as_tuple(self.visit(d, **kwargs) for d in tast.findall('indexRange'))
            if dimensions:
                _type = _type.clone(shape=dimensions)
                variable = variable.clone(dimensions=dimensions)

            if isinstance(_type.dtype, ProcedureType):
                if _type.dtype.name == 'UNKNOWN':
                    # _Probably_ a declaration with implicit interface
                    dtype = ProcedureType(
                        variable.name, is_function=_type.dtype.is_function, return_type=_type.dtype.return_type
                    )
                    _type = _type.clone(dtype=dtype)

                if tast.attrib.get('is_external') == 'true':
                    _type.external = True
                elif variable == kwargs['scope'].name and _type.dtype.return_type is not None:
                    # This is the declaration of the return type inside a function, which is
                    # why we restore the return_type
                    _type = _type.dtype.return_type

        else:
            raise ValueError

        scope = kwargs['scope']
        if o.find('value') is not None:
            _type = _type.clone(initial=AttachScopesMapper()(self.visit(o.find('value'), **kwargs), scope=scope))
        if _type.kind is not None:
            _type = _type.clone(kind=AttachScopesMapper()(_type.kind, scope=scope))

        scope.symbol_attrs[variable.name] = _type
        variable = variable.rescope(scope=scope)

        if isinstance(_type.dtype, ProcedureType):
            # This is actually a function or subroutine (EXTERNAL or PROCEDURE declaration)
            if _type.external:
                return ir.ProcedureDeclaration(symbols=(variable,), external=True, source=kwargs['source'])
            if _type.dtype.name == variable and _type.dtype.is_function:
                return ir.ProcedureDeclaration(
                    symbols=(variable,), interface=_type.dtype.return_type.dtype, source=kwargs['source']
                )
            interface = sym.Variable(name=_type.dtype.name, scope=scope.get_symbol_scope(_type.dtype.name))
            return ir.ProcedureDeclaration(symbols=(variable,), interface=interface, source=kwargs['source'])

        return ir.VariableDeclaration(symbols=(variable,), source=kwargs['source'])

    def visit_FstructDecl(self, o, **kwargs):
        name = o.find('name')
        struct_type = self.type_map[name.attrib['type']]

        # Type attributes
        abstract = struct_type.get('is_abstract') == 'true'
        if 'extends' in struct_type.attrib:
            base_type = kwargs['symbol_map'][struct_type.attrib['extends']]
            extends = base_type.find('name').text
        else:
            extends = None
        bind_c = struct_type.get('bind', '').lower() == 'c'
        private = struct_type.get('is_private', '').lower() == 'true'
        public = struct_type.get('is_public', '').lower() == 'true'

        # Type Parameters
        if struct_type.find('typeParams') is not None:
            self.warn_or_fail('Parameterized types not implemented')

        # Instantiate the TypeDef without its body
        # Note: This creates the symbol table for the declarations and
        # the typedef object registers itself in the parent scope
        typedef = ir.TypeDef(
            name=name.text, body=(), abstract=abstract, extends=extends, bind_c=bind_c,
            private=private, public=public, parent=kwargs['scope'], source=kwargs['source']
        )
        kwargs['scope'] = typedef

        body = []

        # Check if the type is marked as sequence
        if struct_type.get('is_sequence') == 'true':
            body += [ir.Intrinsic('SEQUENCE')]

        # Build the list of derived type members and individual body for each
        if struct_type.find('symbols'):
            variables = self.visit(struct_type.find('symbols'), **kwargs)
            for v in variables:
                if isinstance(v.type.dtype, ProcedureType):
                    if v.type.dtype.name == v and v.type.dtype.is_function:
                        interface = v.type.dtype.return_type
                    else:
                        iface_name = v.type.dtype.name
                        interface = sym.Variable(name=iface_name, scope=kwargs['scope'].get_symbol_scope(iface_name))
                    body += [ir.ProcedureDeclaration(symbols=(v,), interface=interface)]
                else:
                    body += [ir.VariableDeclaration(symbols=(v,))]

        if struct_type.find('typeBoundProcedures'):
            # See if components are marked private
            body += [ir.Intrinsic('CONTAINS')]
            if struct_type.attrib.get('is_internal_private') == 'true':
                body += [ir.Intrinsic('PRIVATE')]
            body += self.visit(struct_type.find('typeBoundProcedures'), **kwargs)

        # Finally: update the typedef with its body
        typedef._update(body=as_tuple(body))
        typedef.rescope_symbols()
        return typedef

    def visit_symbols(self, o, **kwargs):
        """
        Build the list of variables for a `FstructType` node
        """
        variables = []
        for s in o:
            var = self.visit(s.find('name'), **kwargs)
            _type = self.type_from_type_attrib(s.attrib['type'], **kwargs)
            kwargs['scope'].symbol_attrs[var.name] = _type

            if _type.shape:
                var = var.clone(dimensions=_type.shape)
            variables += [var.rescope(scope=kwargs['scope'])]
        return variables

    def visit_typeBoundProcedures(self, o, **kwargs):
        procedures = []
        for i in o:
            proc = self.visit(i, **kwargs)
            if i.get('is_deferred') == 'true':
                assert proc.type.deferred is True
                assert proc.type.bind_names and len(proc.type.bind_names) == 1
                intf = proc.type.bind_names[0]
                procedures += [ir.ProcedureDeclaration(interface=intf, symbols=(proc,))]
            elif i.tag == 'typeBoundGenericProcedure':
                procedures += [ir.ProcedureDeclaration(symbols=(proc,), generic=True)]
            elif i.tag == 'finalProcedure':
                procedures += [ir.ProcedureDeclaration(symbols=(proc,), final=True)]
            else:
                procedures += [ir.ProcedureDeclaration(symbols=(proc,))]
        return procedures

    def visit_typeBoundProcedure(self, o, **kwargs):
        scope = kwargs['scope']
        var = self.visit(o.find('name'), **kwargs)

        _type = self.type_from_type_attrib(o.attrib['type'], **kwargs)
        if o.get('pass') == 'pass':
            _type = _type.clone(pass_attr=o.get('pass_arg_name', True))
        elif o.get('pass') == 'nopass':
            _type = _type.clone(pass_attr=False)
        if o.get('is_deferred') == 'true':
            _type = _type.clone(deferred=True)
        if o.get('is_non_overridable') == 'true':
            _type = _type.clone(non_overridable=True)
        if o.get('is_private') == 'true':
            _type = _type.clone(private=True)
        if o.get('is_public') == 'true':
            _type = _type.clone(public=True)

        if o.find('binding'):
            bind_name = self.visit(o.find('binding/name'), **kwargs)
            bind_name_scope = scope.get_symbol_scope(bind_name.name)

            # Set correct type for interface/binding
            if bind_name_scope is not None:
                bind_name = bind_name.rescope(scope=bind_name_scope)
            else:
                bind_name = bind_name.clone(type=bind_name.type.clone(dtype=ProcedureType(bind_name.name)))

            if bind_name.name.lower() == var.name.lower() and not _type.deferred:
                # No need to assign bind_names property
                _type = _type.clone(dtype=bind_name.type.dtype)
            else:
                # Assign the binding as bind_nameial (and park the interface here for
                # declarations with deferred attribute)
                _type = _type.clone(dtype=bind_name.type.dtype, bind_names=(bind_name,))

        scope.symbol_attrs[var.name] = _type
        return var.rescope(scope=scope)

    def visit_typeBoundGenericProcedure(self, o, **kwargs):
        scope = kwargs['scope']
        var = self.visit(o.find('name'), **kwargs)

        _type = SymbolAttributes(ProcedureType(name=var.name, is_generic=True))
        if o.get('is_private') == 'true':
            _type = _type.clone(private=True)
        if o.get('is_public') == 'true':
            _type = _type.clone(public=True)

        assert o.find('binding') is not None

        bind_names = []
        for name in o.findall('binding/name'):
            bind_name = self.visit(name, **kwargs)
            bind_name_scope = scope.get_symbol_scope(bind_name.name)

            # Set correct type for interface/binding
            if bind_name_scope is not None:
                bind_name = bind_name.rescope(scope=bind_name_scope)
            else:
                bind_name = bind_name.clone(type=bind_name.type.clone(dtype=ProcedureType(bind_name.name)))

            bind_names += [bind_name]

        _type = _type.clone(bind_names=as_tuple(bind_names))
        scope.symbol_attrs[var.name] = _type
        return var.rescope(scope=scope)

    def visit_finalProcedure(self, o, **kwargs):
        scope = kwargs['scope']
        var = self.visit(o.find('name'), **kwargs)
        _type = scope.symbol_attrs.lookup(var.name)
        scope.symbol_attrs[var.name] = _type
        return var.rescope(scope=scope)

    def visit_FdataDecl(self, o, **kwargs):
        variable = self.visit(o.find('varList'), **kwargs)
        values = self.visit(o.find('valueList'), **kwargs)
        return ir.DataDeclaration(variable=variable, values=values, source=kwargs['source'])

    def visit_varList(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o)
        children = tuple(c for c in children if c is not None)
        return children

    visit_valueList = visit_varList

    def visit_FbasicType(self, o, **kwargs):
        ref = o.attrib.get('ref', None)
        if ref in self._omni_types:
            dtype = BasicType.from_fortran_type(self._omni_types[ref])
            kind = self.visit(o.find('kind'), **kwargs) if o.find('kind') is not None else None
            length = o.find('len')
            if length is not None:
                if length == '*':
                    pass
                elif length.attrib.get('is_assumed_size') == 'true':
                    length = '*'
                elif length.attrib.get('is_assumed_shape') == 'true':
                    length = ':'
                else:
                    length = self.visit(length, **kwargs)
            _type = SymbolAttributes(dtype, kind=kind, length=length)
        elif ref in self.type_map:
            if o.find('name') is not None:
                _type = self.visit(self.type_map[ref], name=o.find('name').text, **kwargs)
            else:
                _type = self.visit(self.type_map[ref], **kwargs)
            if o.attrib.get('is_class') == 'true':
                _type = _type.clone(polymorphic=True)
        elif ref == 'FnumericAll':
            _type = SymbolAttributes(BasicType.DEFERRED)
        else:
            raise ValueError

        shape = o.findall('indexRange')
        if shape:
            _type.shape = tuple(self.visit(s, **kwargs) for s in shape)

        # OMNI types are build recursively from references (Matroshka-style)
        if o.get('intent') is not None:
            _type.intent = o.get('intent')
        if o.get('is_allocatable') == 'true':
            _type.allocatable = True
        if o.get('is_pointer') == 'true':
            _type.pointer = True
        if o.get('is_optional') == 'true':
            _type.optional = True
        if o.get('is_parameter') == 'true':
            _type.parameter = True
        if o.get('is_target') == 'true':
            _type.target = True
        if o.get('is_contiguous') == 'true':
            _type.contiguous = True
        if o.get('is_private') == 'true':
            _type.private = True
        if o.get('is_public') == 'true':
            _type.public = True
        return _type

    def visit_FfunctionType(self, o, **kwargs):
        if o.attrib['return_type'] == 'Fvoid':
            return_type = None
        elif o.attrib['return_type'] in self._omni_types:
            return_type = SymbolAttributes(BasicType.from_fortran_type(self._omni_types[o.attrib['return_type']]))
        elif o.attrib['return_type'] in self.type_map:
            return_type = self.visit(self.type_map[o.attrib['return_type']], **kwargs)
        else:
            raise ValueError

        if o.attrib['type'] in kwargs['symbol_map']:
            name = kwargs['symbol_map'][o.attrib['type']].find('name').text
        else:
            name = kwargs.get('name', 'UNKNOWN')
        dtype = ProcedureType(name, is_function=return_type is not None, return_type=return_type)

        prefix = []
        if o.attrib.get('is_pure') == 'true':
            prefix += ['PURE']
        if o.attrib.get('is_elemental') == 'true':
            prefix += ['ELEMENTAL']
        return SymbolAttributes(dtype, prefix=prefix or None)

    def visit_FstructType(self, o, **kwargs):
        # We have encountered a derived type as part of the declaration in the spec
        # of a routine.
        name = o.attrib['type']
        if name in kwargs['symbol_map']:
            name = kwargs['symbol_map'][name].find('name').text

        # Check if we know that type already
        dtype = kwargs['scope'].symbol_attrs.lookup(name, recursive=True)
        if dtype is None or dtype.dtype == BasicType.DEFERRED:
            dtype = DerivedType(name=name, typedef=BasicType.DEFERRED)
        else:
            dtype = dtype.dtype

        return SymbolAttributes(dtype)

    def visit_value(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    visit_kind = visit_value
    visit_len = visit_value

    def visit_associateStatement(self, o, **kwargs):
        associations = tuple(self.visit(c, **kwargs) for c in o.findall('symbols/id'))

        # Create a scope for the associate
        parent_scope = kwargs['scope']
        associate = ir.Associate(associations=(), body=(), parent=parent_scope, source=kwargs['source'])
        kwargs['scope'] = associate

        # Put associate expressions into the right scope and determine type of new symbols
        rescoped_associations = []
        for expr, name in associations:
            # Put symbols in associated expression into the right scope
            expr = AttachScopesMapper()(expr, scope=parent_scope)

            # Determine type of new names
            if isinstance(expr, (sym.TypedSymbol, sym.MetaSymbol)):
                # Use the type of the associated variable
                _type = expr.type.clone(parent=None)
                if isinstance(expr, sym.Array) and expr.dimensions is not None:
                    shape = ExpressionDimensionsMapper()(expr)
                    if shape == (sym.IntLiteral(1),):
                        # For a scalar expression, we remove the shape
                        shape = None
                    _type = _type.clone(shape=shape)
            else:
                # TODO: Handle data type and shape of complex expressions
                shape = ExpressionDimensionsMapper()(expr)
                if shape == (sym.IntLiteral(1),):
                    # For a scalar expression, we remove the shape
                    shape = None
                _type = SymbolAttributes(BasicType.DEFERRED, shape=shape)
            name = name.clone(scope=associate, type=_type)
            rescoped_associations += [(expr, name)]
        associations = as_tuple(rescoped_associations)

        body = self.visit(o.find('body'), **kwargs)
        associate._update(associations=associations, body=body)
        return associate

    def visit_id(self, o, **kwargs):
        expr = self.visit(o.find('value'), **kwargs)
        name = self.visit(o.find('name'), **kwargs)
        return expr, name

    def visit_exprStatement(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_FcommentLine(self, o, **kwargs):
        return ir.Comment(text=o.text, source=kwargs['source'])

    def visit_FpragmaStatement(self, o, **kwargs):
        keyword = o.text.split(' ')[0]
        content = ' '.join(o.text.split(' ')[1:])
        return ir.Pragma(keyword=keyword, content=content, source=kwargs['source'])

    def visit_FassignStatement(self, o, **kwargs):
        lhs = self.visit(o[0], **kwargs)
        rhs = self.visit(o[1], **kwargs)
        return ir.Assignment(lhs=lhs, rhs=rhs, source=kwargs['source'])

    def visit_FallocateStatement(self, o, **kwargs):
        variables = tuple(self.visit(c, **kwargs) for c in o.findall('alloc'))

        alloc_opts = {}
        if o.find('allocOpt') is not None:
            alloc_opts = [self.visit(opt, **kwargs) for opt in o.findall('allocOpt')]
            alloc_opts = [opt for opt in alloc_opts if opt is not None]
            alloc_opts = dict(alloc_opts)

        return ir.Allocation(variables=variables, source=kwargs['source'],
                             data_source=alloc_opts.get('source'), status_var=alloc_opts.get('stat'))

    def visit_allocOpt(self, o, **kwargs):
        keyword = o.attrib['kind'].lower()
        if keyword in ('source', 'stat'):
            return keyword, self.visit(o[0], **kwargs)
        self.warn_or_fail(f'Unsupported allocation option: {keyword}')
        return None

    def visit_FdeallocateStatement(self, o, **kwargs):
        variables = tuple(self.visit(c, **kwargs) for c in o.findall('alloc'))

        alloc_opts = {}
        if o.find('allocOpt') is not None:
            alloc_opts = [self.visit(opt, **kwargs) for opt in o.findall('allocOpt')]
            alloc_opts = [opt for opt in alloc_opts if opt is not None]
            alloc_opts = dict(alloc_opts)

        return ir.Deallocation(variables=variables, source=kwargs['source'],
                               status_var=alloc_opts.get('stat'))

    def visit_FnullifyStatement(self, o, **kwargs):
        variables = tuple(self.visit(c, **kwargs) for c in o.findall('alloc'))
        return ir.Nullify(variables=variables, source=kwargs['source'])

    def visit_alloc(self, o, **kwargs):
        variable = self.visit(o[0], **kwargs)
        if o.find('arrayIndex') is not None:
            dimensions = tuple(self.visit(c, **kwargs) for c in o.findall('arrayIndex'))
            variable = variable.clone(dimensions=dimensions)
        return variable

    def visit_FwhereStatement(self, o, **kwargs):
        conditions = tuple(self.visit(c, **kwargs) for c in o.findall('condition'))
        bodies = tuple(self.visit(b, **kwargs) for b in o.findall('then/body'))
        if o.find('else') is not None:
            default = self.visit(o.find('else/body'), **kwargs)
        else:
            default = ()
        return ir.MaskedStatement(conditions=conditions, bodies=bodies, default=default, source=kwargs['source'])

    def visit_FpointerAssignStatement(self, o, **kwargs):
        target = self.visit(o[0], **kwargs)
        expr = self.visit(o[1], **kwargs)
        return ir.Assignment(lhs=target, rhs=expr, ptr=True, source=kwargs['source'])

    def visit_FdoWhileStatement(self, o, **kwargs):
        assert o.find('condition') is not None
        assert o.find('body') is not None
        condition = self.visit(o.find('condition'), **kwargs)
        body = self.visit(o.find('body'), **kwargs)
        return ir.WhileLoop(condition=condition, body=body, source=kwargs['source'])

    def visit_FdoStatement(self, o, **kwargs):
        assert o.find('body') is not None
        body = self.visit(o.find('body'), **kwargs)
        if o.find('Var') is None:
            # We are in an unbound do loop
            return ir.WhileLoop(condition=None, body=body, source=kwargs['source'])
        variable = self.visit(o.find('Var'), **kwargs)
        lower = self.visit(o.find('indexRange/lowerBound'), **kwargs)
        upper = self.visit(o.find('indexRange/upperBound'), **kwargs)
        step = self.visit(o.find('indexRange/step'), **kwargs)
        # Drop OMNI's `:1` step counting for ranges in the name of consistency
        step = None if step == '1' else step
        bounds = sym.LoopRange((lower, upper, step), source=kwargs['source'])
        return ir.Loop(variable=variable, body=body, bounds=bounds, source=kwargs['source'])

    def visit_FdoLoop(self, o, **kwargs):
        self.warn_or_fail('implicit do loops not implemented')

    def visit_FifStatement(self, o, **kwargs):
        condition = self.visit(o.find('condition'), **kwargs)
        body = self.visit(o.find('then/body'), **kwargs)
        if o.find('else'):
            else_body = self.visit(o.find('else/body'), **kwargs)
        else:
            else_body = ()
        return ir.Conditional(condition=condition, body=body, else_body=else_body, source=kwargs['source'])

    def visit_condition(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_FselectCaseStatement(self, o, **kwargs):
        expr = self.visit(o.find('value'), **kwargs)
        cases = tuple(self.visit(case, **kwargs) for case in o.findall('FcaseLabel'))
        values, bodies = zip(*cases)
        if None in values:
            else_index = values.index(None)
            else_body = as_tuple(bodies[else_index])
            values = values[:else_index] + values[else_index+1:]
            bodies = bodies[:else_index] + bodies[else_index+1:]
        else:
            else_body = ()

        # Retain comments before the first case
        value_idx, case_idx = list(o).index(o.find('value')), list(o).index(o.find('FcaseLabel'))
        pre = as_tuple(self.visit(c, **kwargs) for c in o[value_idx+1:case_idx])

        return (
            *pre,
            ir.MultiConditional(expr=expr, values=values, bodies=bodies, else_body=else_body,
                                source=kwargs['source'])
        )

    def visit_FcaseLabel(self, o, **kwargs):
        values = [self.visit(value, **kwargs) for value in list(o) if value.tag in ('value', 'indexRange')]
        if not values:
            values = None
        elif len(values) == 1:
            values = values.pop()
        body = self.visit(o.find('body'), **kwargs)
        return as_tuple(values) or None, as_tuple(body)

    def visit_FenumDecl(self, o, **kwargs):
        enum_type = self.type_map[o.attrib['type']]

        # Build the list of symbols
        symbols = []
        for i in enum_type.findall('symbols/id'):
            var = self.visit(i.find('name'), **kwargs)
            initial = i.find('value')
            if initial is not None:
                initial = self.visit(initial, **kwargs)
            _type = SymbolAttributes(BasicType.INTEGER, initial=initial)
            symbols += [var.clone(type=_type)]

        # Put symbols in the right scope (that should register their type in that scope's symbol table)
        symbols = tuple(s.rescope(scope=kwargs['scope']) for s in symbols)

        # Create the enum
        return ir.Enumeration(symbols=symbols, source=kwargs['source'])

    def visit_FmemberRef(self, o, **kwargs):
        parent = self.visit(o.find('varRef'), **kwargs)
        name = f'{parent.name}%{o.attrib["member"]}'
        variable = sym.Variable(name=name, parent=parent)
        return variable

    def visit_name(self, o, **kwargs):
        return sym.Variable(name=o.text, source=kwargs.get('source'))

    visit_Var = visit_name

    def visit_FarrayRef(self, o, **kwargs):
        var = self.visit(o.find('varRef'), **kwargs)
        dimensions = as_tuple(self.visit(i, **kwargs) for i in o[1:])
        var = var.clone(dimensions=dimensions)
        return var

    def visit_varRef(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_arrayIndex(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    def visit_indexRange(self, o, **kwargs):
        lbound = o.find('lowerBound')
        lower = self.visit(lbound, **kwargs) if lbound is not None else None
        ubound = o.find('upperBound')
        upper = self.visit(ubound, **kwargs) if ubound is not None else None
        st = o.find('step')
        step = self.visit(st, **kwargs) if st is not None else None
        # Drop OMNI's `:1` step counting for ranges in the name of consistency
        step = None if step == '1' else step
        return sym.RangeIndex((lower, upper, step), source=kwargs['source'])

    def visit_lowerBound(self, o, **kwargs):
        return self.visit(o[0], **kwargs)

    visit_upperBound = visit_lowerBound
    visit_step = visit_lowerBound

    def visit_FrealConstant(self, o, **kwargs):
        if 'kind' in o.attrib and not 'd' in o.text.lower():
            _type = self.visit(self.type_map[o.attrib.get('type')], **kwargs)
            return sym.Literal(value=o.text, type=BasicType.REAL, kind=_type.kind, source=kwargs['source'])
        return sym.Literal(value=o.text, type=BasicType.REAL, source=kwargs['source'])

    def visit_FlogicalConstant(self, o, **kwargs):
        return sym.Literal(value=o.text, type=BasicType.LOGICAL, source=kwargs['source'])

    def visit_FcharacterConstant(self, o, **kwargs):
        return sym.Literal(value=f'"{o.text}"', type=BasicType.CHARACTER, source=kwargs['source'])

    def visit_FintConstant(self, o, **kwargs):
        if 'kind' in o.attrib:
            _type = self.visit(self.type_map[o.attrib.get('type')], **kwargs)
            return sym.Literal(value=int(o.text), type=BasicType.INTEGER, kind=_type.kind, source=kwargs['source'])
        return sym.Literal(value=int(o.text), type=BasicType.INTEGER, source=kwargs['source'])

    def visit_FcomplexConstant(self, o, **kwargs):
        value = ', '.join(f'{self.visit(v, **kwargs)}' for v in list(o))
        return sym.IntrinsicLiteral(value=f'({value})', source=kwargs['source'])

    def visit_FarrayConstructor(self, o, **kwargs):
        values = as_tuple(self.visit(v, **kwargs) for v in o)
        return sym.LiteralList(values=values)

    def visit_functionCall(self, o, **kwargs):
        if o.find('name') is not None:
            name = self.visit(o.find('name'), **kwargs)
        elif o.find('FmemberRef') is not None:
            name = self.visit(o.find('FmemberRef'), **kwargs)
        else:
            raise ValueError

        args = o.find('arguments')
        if args is not None:
            args = as_tuple(self.visit(a, **kwargs) for a in args)
            # Separate keyword argument from positional arguments
            kw_args = as_tuple(arg for arg in args if isinstance(arg, tuple))
            args = as_tuple(arg for arg in args if not isinstance(arg, tuple))
        else:
            args, kw_args = (), ()

        if o.attrib.get('type', 'Fvoid') == 'Fvoid':
            # Subroutine call
            return ir.CallStatement(name=name, arguments=args, kwarguments=kw_args, source=kwargs['source'])

        if name.name.lower() in ('real', 'int'):
            assert args
            expr = args[0]
            if kw_args:
                assert len(args) == 1
                assert len(kw_args) == 1 and kw_args[0][0] == 'kind'
                kind = kw_args[0][1]
            else:
                kind = args[1] if len(args) > 1 else None
            return sym.Cast(name, expr, kind=kind, source=kwargs['source'])

        return sym.InlineCall(name, parameters=args, kw_parameters=kw_args, source=kwargs['source'])

    def visit_FstructConstructor(self, o, **kwargs):
        _type = self.type_from_type_attrib(o.attrib['type'], **kwargs)
        assert isinstance(_type.dtype, DerivedType)

        name = sym.Variable(name=_type.dtype.name)
        args = [self.visit(a, **kwargs) for a in o]

        # Separate keyword argument from positional arguments
        kw_args = as_tuple(arg for arg in args if isinstance(arg, tuple))
        args = as_tuple(arg for arg in args if not isinstance(arg, tuple))

        return sym.InlineCall(name, parameters=args, kw_parameters=kw_args, source=kwargs['source'])

    def visit_FcycleStatement(self, o, **kwargs):
        # TODO: do-construct-name is not preserved
        return ir.Intrinsic(text='cycle', source=kwargs['source'])

    def visit_continueStatement(self, o, **kwargs):
        return ir.Intrinsic(text='continue', source=kwargs['source'])

    def visit_FexitStatement(self, o, **kwargs):
        # TODO: do-construct-name is not preserved
        return ir.Intrinsic(text='exit', source=kwargs['source'])

    def visit_FopenStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        return ir.Intrinsic(text=f'open({nargs})', source=kwargs['source'])

    def visit_FcloseStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        return ir.Intrinsic(text=f'close({nargs})', source=kwargs['source'])

    def visit_FreadStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        values = [self.visit(v, **kwargs) for v in o.find('valueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        args = ', '.join(f'{v}' for v in values)
        return ir.Intrinsic(text=f'read({nargs}) {args}', source=kwargs['source'])

    def visit_FwriteStatement(self, o, **kwargs):
        nvalues = [self.visit(nv, **kwargs) for nv in o.find('namedValueList')]
        values = [self.visit(v, **kwargs) for v in o.find('valueList')]
        nargs = ', '.join(f'{k}={v}' for k, v in nvalues)
        args = ', '.join(f'{v}' for v in values)
        return ir.Intrinsic(text=f'write({nargs}) {args}', source=kwargs['source'])

    def visit_FprintStatement(self, o, **kwargs):
        values = [self.visit(v, **kwargs) for v in o.find('valueList')]
        args = ', '.join(f'{v}' for v in values)
        fmt = o.attrib['format']
        return ir.Intrinsic(text=f'print {fmt}, {args}', source=kwargs['source'])

    def visit_FformatDecl(self, o, **kwargs):
        fmt = f'FORMAT{o.attrib["format"]}'
        return ir.Intrinsic(text=fmt, source=kwargs['source'])

    def visit_namedValue(self, o, **kwargs):
        name = o.attrib['name']
        if 'value' in o.attrib:
            return name, o.attrib['value']
        return name, self.visit(list(o)[0], **kwargs)

    @staticmethod
    def parenthesize_if_needed(expr, enclosing_cls):
        # Other than FP/OFP, OMNI does not retain any information about parenthesis in the
        # original source. While the parse tree is semantically correct,
        # it may cause problems with some agressively optimising compilers.
        # We inject manual parenthesis here for nested expressions to make sure
        # we capture as much of the evaluation order of the original source as possible.
        # Note: this will result in an abundance of trivial/unnecessary parenthesis!
        if enclosing_cls in (sym.Product, sym.Quotient):
            if isinstance(expr, sym.Product):
                return op.ParenthesisedMul(expr.children, source=expr.source)
            if isinstance(expr, sym.Quotient):
                return op.ParenthesisedDiv(expr.numerator, expr.denominator, source=expr.source)
            if isinstance(expr, sym.Sum):
                return op.ParenthesisedAdd(expr.children, source=expr.source)
            if isinstance(expr, sym.Power):
                return op.ParenthesisedPow(expr.base, expr.exponent, source=expr.source)
        return expr

    def visit_plusExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Sum(exprs, source=kwargs['source'])

    def visit_minusExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Sum((exprs[0], sym.Product((-1, exprs[1]))), source=kwargs['source'])

    def visit_mulExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        exprs = tuple(self.parenthesize_if_needed(c, sym.Product) for c in exprs)
        return sym.Product(exprs, source=kwargs['source'])

    def visit_divExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        exprs = tuple(self.parenthesize_if_needed(c, sym.Quotient) for c in exprs)
        return sym.Quotient(*exprs, source=kwargs['source'])

    def visit_FpowerExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Power(base=exprs[0], exponent=exprs[1], source=kwargs['source'])

    def visit_unaryMinusExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 1
        return sym.Product((-1, exprs[0]), source=kwargs['source'])

    def visit_logOrExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        return sym.LogicalOr(exprs, source=kwargs['source'])

    def visit_logAndExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        return sym.LogicalAnd(exprs, source=kwargs['source'])

    def visit_logNotExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 1
        return sym.LogicalNot(exprs[0], source=kwargs['source'])

    def visit_logLTExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '<', exprs[1], source=kwargs['source'])

    def visit_logLEExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '<=', exprs[1], source=kwargs['source'])

    def visit_logGTExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '>', exprs[1], source=kwargs['source'])

    def visit_logGEExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '>=', exprs[1], source=kwargs['source'])

    def visit_logEQExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '==', exprs[1], source=kwargs['source'])

    def visit_logNEQExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.Comparison(exprs[0], '!=', exprs[1], source=kwargs['source'])

    def visit_logEQVExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.LogicalOr((sym.LogicalAnd(exprs), sym.LogicalNot(sym.LogicalOr(exprs))), source=kwargs['source'])

    def visit_logNEQVExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(exprs)), sym.LogicalOr(exprs)), source=kwargs['source'])

    def visit_FconcatExpr(self, o, **kwargs):
        exprs = tuple(self.visit(c, **kwargs) for c in o)
        assert len(exprs) == 2
        return StringConcat(exprs, source=kwargs['source'])

    def visit_gotoStatement(self, o, **kwargs):
        label = int(o.attrib['label_name'])
        return ir.Intrinsic(text=f'go to {label: d}', source=kwargs['source'])

    def visit_statementLabel(self, o, **kwargs):
        return ir.Comment('__STATEMENT_LABEL__', label=o.attrib['label_name'], source=kwargs['source'])

    def visit_FreturnStatement(self, o, **kwargs):
        return ir.Intrinsic(text='return', source=kwargs['source'])
