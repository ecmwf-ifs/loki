# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.backend.fgen import FortranCodegen

__all__ = ['cufgen', 'CudaFortranCodegen']


class CudaFortranCodegen(FortranCodegen):
    """
    Tree visitor that extends :any:`FortranCodegen` with Cuda Fortran (CUF) language variations.
    """

    def visit_CallStatement(self, o, **kwargs):
        """
        Format call statement as
          CALL(<chevron>) <name>(<args>)
          with the chevron as launch configuration for device offloading,
          resulting in something like
          call kernel<<<grid,block[,bytes][,streamid]>>>(arg1,arg2,...)
        """
        pragma = self.visit(o.pragma, **kwargs)
        name = self.visit(o.name, **kwargs)
        args = self.visit_all(o.arguments, **kwargs)
        if o.chevron is not None:
            chevron = f"<<<{','.join([str(elem) for elem in o.chevron])}>>>"
        else:
            chevron = ""
        if o.kwarguments:
            args += tuple(f'{self.visit(arg[0], **kwargs)}={self.visit(arg[1], **kwargs)}' for arg in o.kwarguments)
        call = self.format_line('CALL ', name, chevron, '(', self.join_items(args), ')')
        return self.join_lines(pragma, call)

    def visit_SymbolAttributes(self, o, **kwargs):
        """
        Format declaration attributes as
          <typename>[(<spec>)] [, <attributes>]
        """
        attr_str = super().visit_SymbolAttributes(o, **kwargs)
        attributes = []

        attr_dic = {
            "device": "DEVICE",
            "managed": "MANAGED",
            "constant": "CONSTANT",
            "shared": "SHARED",
            "pinned": "PINNED",
            "texture": "TEXTURE"
                    }

        for key, value in attr_dic.items():
            if getattr(o, key):
                attributes += [value]

        return self.join_items([attr_str] + attributes)


def cufgen(ir, depth=0, conservative=False, linewidth=132):
    """
    Generate CUDA Fortran code from one or many IR objects/trees.

    Implemented by extending the :class:`FortranCodegen` to support
    CUDA Fortran specific syntax. Refer to the CUDA_FORTRAN_PROGRAMMING_GUIDE_ for more information.

    Supported subset of the CUDA Fortran specifications:

    * variable qualifiers e.g. ``attributes(device)``
    * chevron syntax for to launch kernels e.g. ``call kernel<<<grid,block[,bytes][,streamid]>>>(arg1,arg2,...)``

    Natively supported (via :class:`FortranCodegen`):

    * subroutine/function qualifiers e.g. ``attributes(global)`` via :py:attr:`loki.Subroutine.prefix`
    * kernel loop directives via :class:`loki.ir.Pragma`

    .. _CUDA_FORTRAN_PROGRAMMING_GUIDE: https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/index.html
    """
    return CudaFortranCodegen(depth=depth, linewidth=linewidth, conservative=conservative).visit(ir)
