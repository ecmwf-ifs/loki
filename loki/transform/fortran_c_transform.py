from collections import OrderedDict

from loki.transform.transformation import BasicTransformation
from loki.sourcefile import SourceFile
from loki.backend import fgen, cgen
from loki.ir import (Section, Import, Intrinsic, Interface, Call, Declaration,
                     TypeDef, Statement)
from loki.subroutine import Subroutine
from loki.types import BaseType, DerivedType
from loki.expression import Variable, FindVariables, InlineCall
from loki.visitors import Transformer, FindNodes
from loki.tools import as_tuple


__all__ = ['FortranCTransformation']


class FortranCTransformation(BasicTransformation):
    """
    Fortran-to-C transformation that translates the given routine
    into C and generates the corresponding ISO-C wrappers.
    """

    def _pipeline(self, routine, **kwargs):
        path = kwargs.get('path')

        # Generate Fortran wrapper module
        wrapper = self.generate_iso_c_wrapper(routine)
        self.wrapperpath = (path/routine.name).with_suffix('.c.F90')
        self.write_to_file(wrapper, filename=self.wrapperpath, module_wrap=True)

        # TODO: Invert data/loop accesses from column to row-major
        self.convert_expressions(routine, **kwargs)

        # Generate C source file from Loki IR
        routine.name = '%s_c' % routine.name
        self.c_path = (path/routine.name).with_suffix('.c')
        SourceFile.to_file(source=cgen(routine), path=self.c_path)

    def generate_iso_c_wrapper(self, routine):
        c_structs = self.generate_iso_c_structs_f(routine)
        interface = self.generate_iso_c_interface(routine, c_structs)

        # Generate the wrapper function
        wrapper_spec = Transformer().visit(routine.spec)
        wrapper_spec.prepend(Import(module='iso_c_binding',
                                    symbols=('c_int', 'c_double', 'c_float')))
        for _, td in c_structs.items():
            wrapper_spec.append(td)
        wrapper_spec.append(interface)

        # Create the wrapper function with casts and interface invocation
        local_arg_map = OrderedDict()
        casts_in = []
        casts_out = []
        for arg in routine.arguments:
            if isinstance(arg.type, DerivedType):
                ctype = DerivedType(name=c_structs[arg.name].name, variables=None)
                cvar = Variable(name='%s_c' % arg.name, type=ctype)
                cast_in = InlineCall(name='transfer', arguments=as_tuple(arg),
                                     kwarguments=as_tuple([('mold', cvar)]))
                casts_in += [Statement(target=cvar, expr=cast_in)]

                cast_out = InlineCall(name='transfer', arguments=as_tuple(cvar),
                                      kwarguments=as_tuple([('mold', arg)]))
                casts_out += [Statement(target=arg, expr=cast_out)]
                local_arg_map[arg.name] = cvar

        arguments = [local_arg_map[a] if a in local_arg_map else a for a in routine.argnames]
        wrapper_body = casts_in
        wrapper_body += [Call(name=interface.body[0].name, arguments=arguments)]
        wrapper_body += casts_out
        wrapper = Subroutine(name='%s_C' % routine.name, spec=wrapper_spec, body=wrapper_body)

        # Copy internal argument and declaration definitions
        wrapper.variables = routine.arguments + [v for _, v in local_arg_map.items()]
        wrapper.arguments = routine.arguments
        return wrapper

    def generate_iso_c_structs_f(self, routine):
        """
        Generate the interoperable struct definitions in Fortran.
        """
        structs = OrderedDict()
        for a in routine.arguments:
            if isinstance(a.type, DerivedType):
                decls = []
                for v in a.type.variables:
                    ctype = v.type.dtype.isoctype
                    decls += [Declaration(variables=(v, ), type=ctype)]
                structs[a.name] = TypeDef(name='%s_c' % a.type.name,
                                          bind_c=True, declarations=decls)

        return structs

    def generate_iso_c_interface(self, routine, c_structs):
        """
        Generate the ISO-C subroutine interface
        """
        intf_name = '%s_fc' % routine.name
        isoc_import = Import(module='iso_c_binding',
                             symbols=('c_int', 'c_double', 'c_float'))
        intf_spec = Section(body=as_tuple(isoc_import))
        intf_spec.body += as_tuple(Intrinsic(text='implicit none'))
        intf_spec.body += as_tuple(td for _, td in c_structs.items())
        intf_routine = Subroutine(name=intf_name, spec=intf_spec, args=(),
                                  body=None, bind='%s_c' % routine.name)

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            if isinstance(arg.type, DerivedType):
                ctype = DerivedType(name=c_structs[arg.name].name, variables=None)
            else:
                ctype = arg.type.dtype.isoctype
                ctype.value = arg.dimensions is None or len(arg.dimensions) == 0
            var = Variable(name=arg.name, dimensions=arg.dimensions,
                           shape=arg.shape, type=ctype)
            intf_routine.variables += [var]
            intf_routine.arguments += [var]

        return Interface(body=(intf_routine, ))

    def convert_expressions(self, routine, **kwargs):
        """
        Converts all expressions to C's 0-index, row major format.

        Note: We do not yet apply any index shifting inexpressions,
        meaning we have to rely on the code-generator to insert shifted
        iteration ranges when defining loops.
        """

        # TODO: Take care of the indexing shift between C and Fortran.
        # Basically, we are relying on the CGen to shuft the iteration
        # indices and dearly hope that nobody uses the index's value.
        for v in FindVariables(unique=False).visit(routine.body):
            # Swap index order to row-major
            if v.dimensions is not None and len(v.dimensions) > 0 :
                v.dimensions = as_tuple(reversed(v.dimensions))
