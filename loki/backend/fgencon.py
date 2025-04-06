# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.backend.fgen import FortranCodegen
from loki.frontend.source import SourceStatus


__all__ = ['FortranCodegenConservative']


class FortranCodegenConservative(FortranCodegen):
    """
    Strictly conservative version of :any:`FortranCodegen` visitor
    that will attempt to use existing :any:`Source` information from
    the frontends where possible.
    """

    def visit_Node(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_Node(o, *args, **kwargs)

    def visit_Assignment(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_Assignment(o, *args, **kwargs)

    def visit_CallStatement(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_CallStatement(o, *args, **kwargs)

    def visit_Comment(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            slist = o.source.string.split('!', maxsplit=1)
            pre, txt = (slist[0], slist[1]) if len(slist) > 1 else (None, slist[0])
            if pre and not pre.isspace():
                # An inline comment, only return text and leading whitespace
                ws = ' '*(len(pre)-len(pre.rstrip(' ')))
                return f'{ws}!{txt}'
            return o.source.string
        return super().visit_Comment(o, *args, **kwargs)

    def visit_Conditional(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_Conditional(o, *args, **kwargs)

    def visit_VariableDeclaration(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string

        if o.source and o.source.status == SourceStatus.INVALID_NODE:
            # Attempt to recreate some structure by locating `::`
            slist = o.source.string.split('::', maxsplit=1)
            assert len(slist) == 2

            # If type attributes haven't changed, prefer source
            attributes = str(self.join_items(self._construct_type_attributes(o, **kwargs))) + ' '
            if slist[0].strip() == attributes.strip():
                attributes = slist[0]

            # If declared variables haven't changed, prefer source
            variables = ' ' + str(self.join_items(self._construct_decl_variables(o, **kwargs)))
            if slist[1].strip() == variables.strip():
                variables = slist[1]

            comment = str(self.visit(o.comment, **kwargs)) if o.comment else ''

            return f'{attributes}::{variables}{comment}'

        return super().visit_VariableDeclaration(o, *args, **kwargs)

    def visit_Import(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_Import(o, *args, **kwargs)

    def visit_Loop(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string

        if o.source and o.source.status == SourceStatus.INVALID_CHILDREN:
            # Recapture header and footer from source
            header = o.source.string.splitlines()[0]
            footer = o.source.string.splitlines()[o.source.lines[1]-o.source.lines[0]]

            pragma = self.visit(o.pragma, **kwargs)
            pragma_post = self.visit(o.pragma_post, **kwargs)
            self.depth += self.style.loop_indent
            body = self.visit(o.body, **kwargs)
            self.depth -= self.style.loop_indent
            return self.join_lines(pragma, header, body, footer, pragma_post)

        return super().visit_Loop(o, *args, **kwargs)

    def visit_Section(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_Section(o, *args, **kwargs)

    def visit_Subroutine(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string

        if o.source and o.source.status == SourceStatus.INVALID_CHILDREN:
            # Re-construct header and footer from source if possible
            h_end = o.body.source.lines[0] if o.body.source else o.source.lines[1]
            h_end = min(h_end, o.spec.source.lines[0]) if o.spec.source else h_end
            if o.docstring:
                h_end = min(h_end, o.docstring[0].source.lines[0])

            if h_end < o.source.lines[1]:
                header = '\n'.join(o.source.string.splitlines()[:h_end-o.source.lines[0]])
            else:
                header = self._construct_subroutine_header(o, **kwargs)

            # For one-line footers reconstruct from source
            foot = o.source.string.splitlines()[o.source.lines[1]-o.source.lines[0]]
            if 'END ' in foot.upper():
                footer = foot
            else:
                footer = self._construct_procedure_footer(o, **kwargs)

            self.depth += self.style.procedure_spec_indent
            docstring = self.visit(o.docstring, **kwargs)
            spec = self.visit(o.spec, **kwargs)
            self.depth -= self.style.procedure_spec_indent

            self.depth += self.style.procedure_body_indent
            body = self.visit(o.body, **kwargs)
            self.depth -= self.style.procedure_body_indent

            self.depth += self.style.procedure_contains_indent
            contains = self.visit(o.contains, **kwargs)
            self.depth -= self.style.procedure_contains_indent
            if contains:
                return self.join_lines(header, docstring, spec, body, contains, footer)

            return self.join_lines(header, docstring, spec, body, footer)

        return super().visit_Subroutine(o, *args, **kwargs)

    def visit_Module(self, o, *args, **kwargs):
        if o.source and o.source.status == SourceStatus.VALID:
            return o.source.string
        return super().visit_Module(o, *args, **kwargs)
