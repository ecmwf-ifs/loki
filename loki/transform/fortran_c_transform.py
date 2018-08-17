from loki.transform.transformation import AbstractTransformation
from loki.sourcefile import SourceFile
from loki.backend import fgen, cgen
from loki.ir import Section, Import, Intrinsic, Interface, Call
from loki.subroutine import Subroutine
from loki.types import BaseType
from loki.expression import Variable
from loki.visitors import Transformer
from loki.tools import as_tuple


__all__ = ['FortranCTransformation']


class FortranCTransformation(AbstractTransformation):
    """
    Fortran-to-C transformation that translates the given routine
    into C and generates the corresponding ISO-C wrappers.
    """

    def _pipeline(self, routine, **kwargs):
        path = kwargs.get('path')

        # Generate Fortran wrapper module
        wrapper = self.generate_iso_c_wrapper(routine, suffix='_iso_c')
        self.wrapperpath = (path/wrapper.name).with_suffix('.f90')
        SourceFile.to_file(source=fgen(wrapper), path=self.wrapperpath)

        # TODO: Invert data/loop accesses from column to row-major

        # Generate C source file from Loki IR
        routine.name = '%s_c' % routine.name
        self.c_path = (path/routine.name).with_suffix('.c')
        SourceFile.to_file(source=cgen(routine), path=self.c_path)

    def generate_iso_c_wrapper(self, routine, suffix='_iso_c'):
        kind_c_map = {'real': 'c_double', 'integer': 'c_int', 'logical': 'c_int'}

        # Generate the ISO-C subroutine interface
        intf_name = '%s_fc' % routine.name
        isoc_import = Import(module='iso_c_binding', symbols=('c_int', 'c_double', 'c_ptr'))
        intf_spec = Section(body=as_tuple(isoc_import))
        intf_spec.body += as_tuple(Intrinsic(text='implicit none'))
        intf_routine = Subroutine(name=intf_name, spec=intf_spec, args=(),
                                  body=None, bind='%s_c' % routine.name)

        # Generate variables and types for argument declarations
        for arg in routine.arguments:
            tname = arg.type.name.lower()
            kind = kind_c_map.get(tname, arg.type.kind)
            value = arg.dimensions is None or len(arg.dimensions) == 0
            ctype = BaseType(name=arg.type.name, kind=kind, value=value)
            var = Variable(name=arg.name, dimensions=arg.dimensions,
                           shape=arg.shape, type=ctype)
            intf_routine.variables += [var]
            intf_routine.arguments += [var]
        interface = Interface(body=(intf_routine, ))

        # Generate the wrapper function
        wrapper_spec = Transformer().visit(routine.spec)
        wrapper_spec.append(interface)
        wrapper_body = [Call(name=intf_name, arguments=routine.argnames)]
        wrapper = Subroutine(name='%s%s' % (routine.name, suffix),
                             spec=wrapper_spec, body=wrapper_body)
        # Copy internal argument and declaration definitions
        wrapper.variables = routine.variables
        wrapper.arguments = routine.arguments
        return wrapper
