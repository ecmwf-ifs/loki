"""
Preprocessing utilities for frontends.
"""
from collections import defaultdict, OrderedDict
from pathlib import Path
import io
import re
from pathlib import Path
from collections import defaultdict, OrderedDict
import pcpp

from codetiming import Timer

from loki.logging import debug, perf
from loki.config import config
from loki.tools import as_tuple, gettempdir, filehash
from loki.visitors import FindNodes
from loki.ir import VariableDeclaration, Intrinsic
from loki.frontend.util import OMNI, OFP, FP, REGEX


__all__ = ['preprocess_cpp', 'sanitize_input', 'sanitize_registry', 'PPRule']


def preprocess_cpp(source, filepath=None, includes=None, defines=None):
    """
    Invoke an external C-preprocessor to sanitize input files.

    Note that the global option ``LOKI_CPP_DUMP_FILES`` will cause the intermediate
    preprocessed source to be written to a temporary file in ``LOKI_TMP_DIR``.

    Parameters
    ----------
    source : str
        Source string to preprocess via ``pcpp``
    filepath : str or pathlib.Path
        Optional filepath name, which will be used to derive the filename
        should intermediate file dumping be enabled via the global config.
    includes : (list of) str
        Include paths for the C-preprocessor.
    defines : (list of) str
        Symbol definitions to add to the C-preprocessor.
    """

    class _LokiCPreprocessor(pcpp.Preprocessor):

        def on_comment(self, tok):  # pylint: disable=unused-argument
            # Pass through C-style comments
            return True

        def on_error(self, file, line, msg):
            # Redirect CPP error to our logger and increment return code
            debug(f'[Loki-CPP] {file}:{line: d} error: {msg}')
            self.return_code += 1

    # Add include paths to PP
    pp = _LokiCPreprocessor()
    # Suppress line directives
    pp.line_directive = None

    for i in as_tuple(includes):
        pp.add_path(str(i))

    # Add and sanitize defines to PP
    for d in as_tuple(defines):
        if '=' not in d:
            d += '=1'
        d = d.replace('=', ' ', 1)
        pp.define(d)

    # Parse source through preprocessor
    pp.parse(source)

    if config['cpp-dump-files']:
        if filepath is None:
            pp_path = Path(filehash(source, suffix='.cpp.f90'))
        else:
            pp_path = filepath.with_suffix('.cpp.f90')
        pp_path = gettempdir()/pp_path.name
        debug(f'[Loki] C-preprocessor, writing {str(pp_path)}')

        # Dump preprocessed source to file and read it
        with pp_path.open('w') as f:
            pp.write(f)
        with pp_path.open('r') as f:
            preprocessed = f.read()
        return preprocessed

    # Return the preprocessed string
    s = io.StringIO()
    pp.write(s)
    return s.getvalue()


@Timer(logger=perf, text=lambda s: f'[Loki::Frontend] Executed sanitize_input in {s:.2f}s')
def sanitize_input(source, frontend):
    """
    Apply internal regex-based sanitisation rules to filter out known
    frontend incompatibilities.

    Note that this will create a record of all things stripped
    (``pp_info``), which will be used to re-insert the dropped
    source info when converting the parsed AST to our IR.

    The ``sanitize_registry`` (see below) holds pre-defined rules
    for each frontend.
    """

    # Apply preprocessing rules and store meta-information
    pp_info = OrderedDict()
    for name, rule in sanitize_registry[frontend].items():
        # Apply rule filter over source file
        rule.reset()
        new_source = ''
        for ll, line in enumerate(source.splitlines(keepends=True)):
            ll += 1  # Correct for Fortran counting
            new_source += rule.filter(line, lineno=ll)

        # Store met-information from rule
        pp_info[name] = rule.info
        source = new_source

    return source, pp_info


def reinsert_contiguous(ir, pp_info):
    """
    Reinsert the CONTIGUOUS marker into declaration variables.
    """
    if pp_info:
        for decl in FindNodes(VariableDeclaration).visit(ir):
            if decl._source.lines[0] in pp_info:
                for var in decl.symbols:
                    var.scope.symbol_attrs[var.name] = var.scope.symbol_attrs[var.name].clone(contiguous=True)
    return ir


def reinsert_convert_endian(ir, pp_info):
    """
    Reinsert the CONVERT='BIG_ENDIAN' or CONVERT='LITTLE_ENDIAN' arguments
    into calls to OPEN.
    """
    if pp_info:
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
    if pp_info:
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
    """
    A preprocessing rule that defines and applies a source replacement
    and collects associated meta-data.
    """

    _empty_pattern = re.compile('')

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


sanitize_registry = {
    REGEX: {},
    OMNI: {},
    OFP: {
        # Remove various IBM directives
        'IBM_DIRECTIVES': PPRule(match=re.compile(r'(@PROCESS.*\n)'), replace='\n'),

        # Despite F2008 compatability, OFP does not recognise the CONTIGUOUS keyword :(
        'CONTIGUOUS': PPRule(
            match=re.compile(r', CONTIGUOUS', re.I), replace='', postprocess=reinsert_contiguous),

        # Strip line annotations from Fypp preprocessor
        'FYPP ANNOTATIONS': PPRule(match=re.compile(r'(# [1-9].*\".*\.fypp\"\n)'), replace=''),
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
            replace=lambda m: m['pp'] or f'"{m["else"]}"'),

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
            replace=lambda m: f'{m["ws"]}{m["open"]}{m["newunit_val"]}{m["delim"] or ""}' +
                              f'{m["args1"]}{m["args2"]}',
            postprocess=reinsert_open_newunit),

        # Strip line annotations from Fypp preprocessor
        'FYPP ANNOTATIONS': PPRule(match=re.compile(r'(# [1-9].*\".*\.fypp\"\n)'), replace=''),
    }
}
"""
The frontend sanitization registry dict holds workaround rules for
Fortran features that cause bugs and failures in frontends. It's
mostly a regex expression that removes certains strings and stores
them, so that they can be re-inserted into the IR by a callback.
"""
