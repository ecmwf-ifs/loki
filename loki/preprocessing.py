import re
from collections import defaultdict

from loki.expression import ExpressionVisitor
from loki.visitors import FindNodes
from loki.ir import Declaration, Statement, Conditional, CommentBlock


__all__ = ['blacklist', 'PPRule']


class InsertLiteralKinds(ExpressionVisitor):
    """
    Re-insert explicit _KIND casts for literals dropped during pre-processing.

    :param pp_info: List of `(literal, kind)` tuples to be inserted
    """

    def __init__(self, pp_info):
        super(InsertLiteralKinds, self).__init__()

        self.pp_info = dict(pp_info)

    def visit_Literal(self, o):
        if o._source.lines[0] in self.pp_info:
            info = self.pp_info[o._source.lines[0]]
            literals = {i['number']: i['kind'] for i in info}
            if o.value in literals:
                o.value = '%s_%s' % (o.value, literals[o.value])

    def visit_CommentBlock(self, o):
        for c in o.comments:
            self.visit(c)

    def visit_Comment(self, o):
        if o._source.lines[0] in self.pp_info:
            info = self.pp_info[o._source.lines[0]]
            for i in info:
                val = i['number']
                kind = i['kind']
                o._source.string = o._source.string.replace(val, '%s_%s' % (val, kind))


def reinsert_literal_kinds(ir, pp_info):
    """
    Re-insert literal _KIND type casts from pre-processing info Note,
    that this is needed to get accurate data _KIND attributes for
    literal values, as these have been stripped in a preprocessing
    step to avoid OFP bugs.
    """
    if pp_info is not None:
        insert_kind = InsertLiteralKinds(pp_info)

        for decl in FindNodes(Declaration).visit(ir):
            for v in decl.variables:
                if v.initial is not None:
                    insert_kind.visit(v.initial)

        for stmt in FindNodes(Statement).visit(ir):
            insert_kind.visit(stmt)

        for cnd in FindNodes(Conditional).visit(ir):
            for c in cnd.conditions:
                insert_kind.visit(c)

        for cmt in FindNodes(CommentBlock).visit(ir):
            insert_kind.visit(cmt)

    return ir


def reinsert_contiguous(ir, pp_info):
    """
    Reinsert the CONTIGUOUS marker into declaration variables.
    """
    if pp_info is not None:
        for decl in FindNodes(Declaration).visit(ir):
            if decl._source.lines[0] in pp_info:
                for v in decl.variables:
                    v.type.contiguous = True
    return ir


class PPRule(object):

    _empty_pattern = re.compile('')

    """
    A preprocessing rule that defines and applies a source replacement
    and collects according meta-data.
    """
    def __init__(self, match, replace, postprocess=None):
        self.match = match
        self.replace = replace

        self._postprocess = postprocess
        self._info = defaultdict(list)

    def filter(self, line, lineno):
        """
        Filter a source line by matching the given rule and storing meta-content.
        """
        if isinstance(self.match, type(self._empty_pattern)):
            # Apply a regex pattern to the line and return 'all'
            for info in self.match.finditer(line):
                self._info[lineno] += [info.groupdict()]
            return self.match.sub(self.replace, line)
        else:
            # Apply a regular string replacement
            if self.match in line:
                self._info[lineno] += [(self.match, self.replace)]
            return line.replace(self.match, self.replace)

    @property
    def info(self):
        """
        Meta-information that will be dumped alongside preprocessed source
        files to re-insert information into a fully parsed IR tree.
        """
        return self._info

    def postprocess(self, ir, info):
        if self._postprocess is not None:
            return self._postprocess(ir, info)
        else:
            return ir


"""
A black list of Fortran features that cause bugs and failures in the OFP.
"""
blacklist = {
    # Remove various IBM directives
    'IBM_NOCHECK': PPRule(match='@PROCESS NOCHECK', replace=''),
    'IBM_HOT': PPRule(match='@PROCESS HOT(NOVECTOR) NOSTRICT', replace=''),

    # Strip and re-insert _KIND type casts to circumvent OFP bug (issue #4)
    'KIND_JPRB': PPRule(match=re.compile('(?P<all>(?P<number>[0-9.]+[eE]?[0-9\-]*)_(?P<kind>JPRB))'),
                        replace=lambda m: m.groupdict()['number'], postprocess=reinsert_literal_kinds),
    'KIND_JPIM': PPRule(match=re.compile('(?P<all>(?P<number>[0-9.]+[eE]?[0-9\-]*)_(?P<kind>JPIM))'),
                        replace=lambda m: m.groupdict()['number'], postprocess=reinsert_literal_kinds),

    # Despit F2008 compatability, OFP does not recognise the CONTIGUOUS keyword :(
    'CONTIGUOUS' : PPRule(match=', CONTIGUOUS', replace='', postprocess=reinsert_contiguous),
}
