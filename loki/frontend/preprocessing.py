import re
from collections import defaultdict

from loki.visitors import FindNodes
from loki.ir import Declaration


__all__ = ['blacklist', 'PPRule']


def reinsert_contiguous(ir, pp_info):
    """
    Reinsert the CONTIGUOUS marker into declaration variables.
    """
    if pp_info is not None:
        for decl in FindNodes(Declaration).visit(ir):
            if decl._source.lines[0] in pp_info:
                decl.type.contiguous = True
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

    def reset(self):
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
    'IBM_DIRECTIVES': PPRule(match=re.compile('(@PROCESS.*\n)'), replace='\n'),

    # Despite F2008 compatability, OFP does not recognise the CONTIGUOUS keyword :(
    'CONTIGUOUS': PPRule(match=', CONTIGUOUS', replace='', postprocess=reinsert_contiguous),
}
