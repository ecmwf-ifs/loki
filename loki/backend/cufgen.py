from loki.backend.fgen import FortranCodegen

__all__ = ['cufgen', 'CudaFortranCodegen']


class CudaFortranCodegen(FortranCodegen):
    """
    Tree visitor that extends `FortranCodegen` with Cuda Fortran (CUF) language variations.
    """

    def __init__(self, depth=0, indent='  ', linewidth=90, conservative=True):
        super().__init__(depth=depth, indent=indent, linewidth=linewidth, conservative=conservative)

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
    Generate standardized Fortran code from one or many IR objects/trees.
    """
    return CudaFortranCodegen(depth=depth, linewidth=linewidth, conservative=conservative).visit(ir)
