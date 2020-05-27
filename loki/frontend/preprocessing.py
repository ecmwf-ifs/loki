import re
from collections import defaultdict

from loki.visitors import FindNodes
from loki.ir import Declaration, Intrinsic
from loki.frontend.util import OMNI, OFP, FP


__all__ = ['blacklist', 'PPRule']


def reinsert_contiguous(ir, pp_info):
    """
    Reinsert the CONTIGUOUS marker into declaration variables.
    """
    if pp_info is not None:
        for decl in FindNodes(Declaration).visit(ir):
            if decl._source.lines[0] in pp_info:
                for var in decl.variables:
                    var.type.contiguous = True
    return ir


def reinsert_convert_endian(ir, pp_info):
    """
    Reinsert the CONVERT='BIG_ENDIAN' or CONVERT='LITTLE_ENDIAN' arguments
    into calls to OPEN.
    """
    if pp_info is not None:
        for intr in FindNodes(Intrinsic).visit(ir):
            match = pp_info.get(intr._source.lines[0], None)
            if match is not None:
                match = match[0]
                intr.text = match['pre'] + match['convert'] + match['post']
                if intr._source is not None:
                    intr._source.string = intr.text
    return ir


class PPRule:

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
        return ir


"""
A black list of Fortran features that cause bugs and failures in frontends.
"""
blacklist = {
    OMNI: {},
    OFP: {
        # Remove various IBM directives
        'IBM_DIRECTIVES': PPRule(match=re.compile(r'(@PROCESS.*\n)'), replace='\n'),

        # Despite F2008 compatability, OFP does not recognise the CONTIGUOUS keyword :(
        'CONTIGUOUS': PPRule(
            match=re.compile(r', CONTIGUOUS', re.I), replace='', postprocess=reinsert_contiguous),
    },
    FP: {
        # Remove various IBM directives
        'IBM_DIRECTIVES': PPRule(match=re.compile(r'(@PROCESS.*\n)'), replace='\n'),

        # Replace string CPP directives in Fortran source lines by strings
        'STRING_PP_DIRECTIVES': PPRule(
            match=re.compile(r'(^(?!\s*#).*)(__(?:FILE(?:NAME)?|DATE|VERSION)__)'),
            replace=r'\1"\2"'),

        # Replace integer CPP directives by 0
        'INTEGER_PP_DIRECTIVES': PPRule(match='__LINE__', replace='0'),

        # Replace CONVERT argument in OPEN calls
        'CONVERT_ENDIAN': PPRule(
            match=re.compile((r'(?:^\s*)(?P<pre>OPEN\s*\(.*)'
                              r'(?P<convert>,\s*CONVERT=[\'\"](?:BIG|LITTLE)_ENDIAN[\'\"]\s*)'
                              r'(?P<post>(?:,.*)?\))'), re.I),
            replace=r'\g<pre>\g<post>', postprocess=reinsert_convert_endian),
    }
}
