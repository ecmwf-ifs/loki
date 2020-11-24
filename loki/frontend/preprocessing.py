import io
import re
import pcpp
import pickle
from collections import defaultdict, OrderedDict

from loki.logging import debug, DEBUG
from loki.config import config
from loki.tools import as_tuple, gettempdir, filehash, timeit
from loki.visitors import FindNodes
from loki.ir import Declaration, Intrinsic
from loki.frontend.util import OMNI, OFP, FP


__all__ = ['preprocess_cpp', 'sanitize_input', 'sanitize_registry', 'PPRule']


def preprocess_cpp(source, filepath=None, includes=None, defines=None):
    """
    Invoke an external C-preprocessor to sanitize input files.

    Note that the global option ``LOKI_CPP_DUMP_FILES`` will cause the intermediate
    preprocessed source to be written to a temporary file in ``LOKI_TMP_DIR``.

    Parameters:
    ===========
    * ``source``: Source string to preprocess via ``pcpp``
    * ``filepath``: Optional filepath name, which will be used to derive the filename
                    should intermediate file dumping be enabled via the global config.
    * ``includes``: (List of) include paths to add to the C-preprocessor.
    * ``defines``: (List of) symbol definitions to add to the C-preprocessor.
    """

    class _LokiCPreprocessor(pcpp.Preprocessor):

        def on_error(self, file, line, msg):
            # Redict CPP error to our logger and increment return code
            debug("[Lok-CPP] %s:%d error: %s" % (file, line, msg))
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
        debug("[Loki] C-preprocessor, writing {}".format(str(pp_path)))

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


@timeit(log_level=DEBUG, getter=lambda x: '' if 'filepath' not in x else x['filepath'].stem)
def sanitize_input(source, frontend, filepath=None):
    """
    Apply internal regex-based sanitisation rules to filter out known
    frontend incompatibilities.

    Note that this will create a record of all things stripped
    (``pp_info``), which will be used to re-insert the dropped
    source info when converting the parsed AST to our IR.

    The ``sanitize_registry`` (see below) holds pre-defined rules
    for each frontend.
    """
    tmpdir = gettempdir()
    pp_path = filepath.with_suffix('.{}{}'.format(str(frontend), filepath.suffix))
    pp_path = tmpdir/pp_path.name
    info_path = filepath.with_suffix('.{}.info'.format(str(frontend)))
    info_path = tmpdir/info_path.name

    debug("[Loki] Pre-processing source file {}".format(str(filepath)))

    # Check for previous preprocessing of this file
    if config['frontend-pp-cache'] and pp_path.exists() and info_path.exists():
        # Make sure the existing PP data belongs to this file
        if pp_path.stat().st_mtime > filepath.stat().st_mtime:
            debug("[Loki] Frontend preprocessor, reading {}".format(str(info_path)))
            with info_path.open('rb') as f:
                pp_info = pickle.load(f)
            if pp_info.get('original_file_path') == str(filepath):
                # Already pre-processed this one,
                # return the cached info and source.
                debug("[Loki] Frontend preprocessor, reading {}".format(str(pp_path)))
                with pp_path.open() as f:
                    source = f.read()
                return source, pp_info

    # Apply preprocessing rules and store meta-information
    pp_info = OrderedDict()
    pp_info['original_file_path'] = str(filepath)
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

    if config['frontend-pp-cache']:
        # Write out the preprocessed source and according info file
        debug("[Loki] Frontend preprocessor, storing {}".format(str(pp_path)))
        with pp_path.open('w') as f:
            f.write(source)
        debug("[Loki] Frontend preprocessor, storing {}".format(str(info_path)))
        with info_path.open('wb') as f:
            pickle.dump(pp_info, f)

    return source, pp_info


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
sanitize_registry = {
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
