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
            match = pp_info.get(intr._source.lines[0], [None])[0]
            if match is not None:
                source = intr._source
                assert source is not None
                text = match['ws'] + match['pre'] + match['convert'] + match['post']
                if match['post'].rstrip().endswith('&'):
                    cont_line_index = source.string.find(match['post']) + len(match['post'])
                    text += source.string[cont_line_index:].rstrip()
                source.string = text
                intr._update(text=text, source=source)
    return ir


def reinsert_open_newunit(ir, pp_info):
    """
    Reinsert the NEWUNIT=... arguments into calls to OPEN.
    """
    if pp_info is not None:
        for intr in FindNodes(Intrinsic).visit(ir):
            match = pp_info.get(intr._source.lines[0], [None])[0]
            if match is not None:
                source = intr._source
                assert source is not None
                text = match['ws'] + match['open'] + match['args1'] + (match['delim'] or '')
                text += match['newunit_key'] + match['newunit_val'] + match['args2']
                if match['args2'].rstrip().endswith('&'):
                    cont_line_index = source.string.find(match['args2']) + len(match['args2'])
                    text += source.string[cont_line_index:].rstrip()
                source.string = text
                intr._update(text=text, source=source)
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

        # Enquote string CPP directives in Fortran source lines to make them string constants
        # Note: this is a bit tricky as we need to make sure that we don't replace it inside CPP
        #       directives as this can produce invalid code
        'STRING_PP_DIRECTIVES': PPRule(
            match=re.compile((
                r'(?P<pp>^\s*#.*__(?:FILE|FILENAME|DATE|VERSION)__)|'  # Match inside a directive
                r'(?P<else>__(?:FILE|FILENAME|DATE|VERSION)__)')),     # Match elsewhere
            replace=lambda m: m['pp'] or '"{}"'.format(m['else'])),

        # Replace integer CPP directives by 0
        'INTEGER_PP_DIRECTIVES': PPRule(match='__LINE__', replace='0'),

        # Replace CONVERT argument in OPEN calls
        'CONVERT_ENDIAN': PPRule(
            match=re.compile((r'(?P<ws>^\s*)(?P<pre>OPEN\s*\(.*?)'
                              r'(?P<convert>,?\s*CONVERT=[\'\"](?:BIG|LITTLE)_ENDIAN[\'\"]\s*)'
                              r'(?P<post>.*?$)'), re.I),
            replace=r'\g<ws>\g<pre>\g<post>', postprocess=reinsert_convert_endian),

        # Replace NEWUNIT argument in OPEN calls
        'OPEN_NEWUNIT': PPRule(
            match=re.compile((r'(?P<ws>^\s*)(?P<open>OPEN\s*\()(?P<args1>.*?)(?P<delim>,)?'
                              r'(?P<newunit_key>,?\s*NEWUNIT=)(?P<newunit_val>.*?(?=,|\)|&))'
                              r'(?P<args2>.*?$)'), re.I),
            replace=lambda m: '{ws}{op}{unit}{delim}{args1}{args2}'.format(
                ws=m['ws'], op=m['open'], unit=m['newunit_val'], delim=m['delim'] or '',
                args1=m['args1'], args2=m['args2']),
            postprocess=reinsert_open_newunit),
    }
}
