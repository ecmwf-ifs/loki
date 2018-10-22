from pathlib import Path
from tempfile import gettempdir
import click
import shutil

from loki.build import (Lib, Builder, clean, GNUCompiler, execute,
                        delete, default_logger, FileLogger, DEBUG, INFO)

# Pre-define static paths and filenames
prec = 'dp'
inproot = Path('/scratch/rd/naml/RAPS/17/inputs')
outroot = Path('/scratch/rd/naml/rundir/RAPS/17')

# TODO: Figure out rootdir in a nocer way
rootdir = Path.cwd().parent
bmdir = rootdir/'flexbuild'
intfbroot = bmdir/'raps17/intfb'

compsys = 'gnu.melchior'
tmpdir = Path(gettempdir())
dumdir = tmpdir/('tmpdir.%s.%s' % (compsys, prec))

source_dirs = [rootdir/'ifs', rootdir/'ifsaux', rootdir/'odb']

###############################################################

incdirs = [
    'algor', 'algor/interface', 'bl', 'bl/include', 'ecfftw',
    'ecfftw/module', 'flexbuild/raps17/intfb',
    'flexbuild/raps17/intfb/ifs', 'flexbuild/raps17/intfb/satrad',
    'ifs', 'ifs/ald_inc', 'ifs/ald_inc/function',
    'ifs/ald_inc/interface', 'ifs/ald_inc/namelist', 'ifs/common',
    'ifs/function', 'ifs/interpol', 'ifs/namelist', 'ifsaux',
    'ifsaux/eclite', 'ifsaux/fa', 'ifsaux/fi_libc',
    'ifsaux/fi_pthread', 'ifsaux/include', 'ifsaux/lfi',
    'ifsaux/lfi_alt', 'ifsaux/support', 'ifsaux/svipc',
    'ifsaux/svipc/include', 'ifsobs/src', 'ifsobs/src/include', 'odb',
    'odb/include', 'odb/interface', 'oopsifs_unboosted/src',
    'oopsifs_unboosted/src/ifs', 'radiation', 'radiation/module',
    'satrad', 'satrad/emiss', 'satrad/include', 'satrad/interface',
    'surf', 'surf/function', 'surf/interface', 'trans',
    'trans/interface', 'wam', 'wam/Alt', 'wam/Alt/Include',
    'wam/Wam_parameter', 'wam/interface', 'wam/module',
]

defs = [
    'LITTLE_ENDIAN', 'BLAS', 'LITTLE', 'LINUX', 'INTEGER_IS_INT',
    '_ABI64', 'F90', 'PARAL', 'NONCRAYF', '_RTTOV_DO_DISTRIBCOEF',
    'NO_CURSES', 'WITH_WAVE', 'WITH_FFTW', 'WITH_ATLAS', 'WITH_FCKIT',
    'HAVE_ECCODES', 'HAVE_NETCDF', 'HAVE_HDF5', 'ECMWF'
]
defs = ['-D%s' % d for d in defs]

undefs = ['ODB_API_SUPPORT', 'WITH_OASIS', 'CANARI', 'HAVE_ODB2']
undefs = ['-U%s' % d for d in undefs]

# Double precision build
undefs += ['-UPARKIND1_SINGLE']


###############################################################
#  Define the NEMO sub-package build
#  TODO: This is currently still just a conceptual draft.

# A dedicated separate logger would be nice....
# TODO: Custom logger object needs thread-listener syncing for parallel builds

# nemologger = FileLogger(name='NEMO-build', filename=bmdir/'nemo_build.log', level=WARNING)
# nemolib = Lib(name='nemo.ORCA.whatever', objs=[])#, logger=nemologger)

# Define various other build components
# def build_pigz():
#     execute(bmdir/'makebins/build_pigz')

# pigz = Binary(name=bmdir/'bin/pigz')  #, callback=build_pigz)

# # Build the dummies objects using build customization via callbacks
# def generate_dummies():
#     """
#     Generate dummies for missing symbols by letting a link attempt
#     fail and passing the log to the create_dummies script.
#     """
#     pass

# TODO: Do we really need dummies, and if so, can we figure out which
# ones without having to do a failing build/link?
# dummyobj = Obj(name=dumdir/'dummies.c')  #, callback=generate_dummies)

######################################

def build_odb():
    """
    Super-custom way of building sub-packages...
    """
    tmpdir = Path(gettempdir())
    odbdir = rootdir/'odb/compiler'

    for ext in ['c', 'h', 'l', 'y']:

        for f in odbdir.glob('*.%s' % ext):
            shutil.copy(str(f), str(tmpdir))

    execute(['bison', '--yacc', '-d', 'yacc.y'], cwd=tmpdir)
    execute(['flex', 'lex.l'], cwd=tmpdir)


################################
#  Define component libraries  #

ignore = [
    'scripts/**/*', 'offline/**/*', 'make/**/*', 'build/**/*',
    'ddl/**/*', 'bin/**/*', 'cma/**/*', 'compiler/**/*', 'dummy/**/*',
    'programs/**/*', 'examples/**/*', 'doc/**/*', 'data/**/*',
    'not_used/**/*', 'sms/**/*', 'perl/**/*', 'docs/**/*',
    'tests/**/*', 'cmake/**/*', 'bamboo/**/*', 'contrib/**/*',
    'sandbox/**/*',
]

pattern = ['*.F90', '*.F', '*.c', '*.cc']

ifsaux_ignore = [
    'programs/**/*', 'ddh/**/*', 'grib_mf/**/*', 'misc/**/*', 'cma/**/*',
    'py_interface/**/*', 'mpi4to8_s.F90', 'mpi4to8.F90',
    'mpi4to8_m.F90', 'opfla_perfmon.c'
]
ifsaux = Lib(name='ifsaux', shared=False, source_dir=rootdir/'ifsaux',
             pattern=pattern, ignore=ifsaux_ignore + ignore)

algor = Lib(name='algor', shared=False, source_dir=rootdir/'algor', pattern=pattern)
bl = Lib(name='bl', shared=False, source_dir=rootdir/'bl', pattern=pattern)
ecfftw = Lib(name='ecfftw', shared=False, source_dir=rootdir/'ecfftw', pattern=pattern)
enkf = Lib(name='enkf', shared=False, source_dir=rootdir/'enkf', pattern=pattern)
ifsobs = Lib(name='ifsobs', shared=False, source_dir=rootdir/'ifsobs', pattern=pattern)
crm = Lib(name='crm', shared=False, source_dir=rootdir/'crm', pattern=pattern)
ifs = Lib(name='ifs', shared=False, source_dir=rootdir/'ifs',
          pattern=pattern, ignore=ignore)

odb_ignore = [
    'extras/**/*', 'tools/**/*', 'pandor/**/*', 'bufr2odb/**/*', 'ddl/**/*',
    'hdr_aligned_tables.h', 'b2o_*.F90', 'b2o_*.h'
]
odb = Lib(name='odb', shared=False, source_dir=rootdir/'odb',
          pattern=pattern, ignore=odb_ignore + ignore)

radiation = Lib(name='radiation', shared=False, source_dir=rootdir/'radiation', pattern=pattern)
satrad = Lib(name='satrad', shared=False, source_dir=rootdir/'satrad', pattern=pattern)
surf = Lib(name='surf', shared=False, source_dir=rootdir/'surf',
           pattern=pattern, ignore=ignore)
trans = Lib(name='trans', shared=False, source_dir=rootdir/'trans', pattern=pattern)

wam_ignore = [
    'Buoy/**/*', 'Wam_obsoletey/**/*', 'Wam_oper_setup/**/*', 'Wam_others/**/*',
    'Wam_setup/**/*', 'Wam_doc/**/*', 'gpl/**/*',
    'alt_hist_prep.F', 'ave.F', 'axis.F', 'axis_plotting.F', 'cab.F',
    'cabqms.F', 'cl_magcl.F', 'colura.F', 'colwnd.F',
    'contour_plotting.F', 'curvef.F', 'defsp.F', 'desgrib.F',
    'esf_preprocessor.F', 'extcbm.F', 'firbo.F', 'graph_plotting.F',
    'hist_draw.F', 'hist_ini.F', 'indlole.F', 'legend_plotting.Fljbs.F',
    'moment.F', 'pfsinp.F', 'pl_bar.F', 'pl_grid.F', 'pl_grid_main.F',
    'plegst.F', 'playout.F', 'plmesy.F', 'plot_layout.F', 'plotdat.F',
    'plotdat.F', 'plothea.F', 'plotnu.F', 'plotsy.F', 'plotte.F',
    'pltcontw.F', 'pltetr.F', 'pltinp.F', 'pltmeanc.F', 'pltura.F',
    'pserie.F', 'psetm.F', 'qlrwd_prep.F', 'sarscat.F', 'scatin.F',
    'scor.F', 'secbo.F', 'show_esf.F', 'statse_xONy.F', 'statsp_xONy.F',
    'text_plotting.F', 'textbox.F', 'titmak.F', 'u10.F', 'ura_col_prep.F',
    'uracal.F', 'uracol.F', 'uraetr.F', 'urapfs.F', 'urapld.F',
    'uraplt.F', 'uraplt2.F', 'urapltm.F', 'trocal.F', 'trocol.F',
    'coltro.F', 'decplp.F', 'mask_wind.F', 'wam_hist_prep.F', 'inmarsb.F',
    'inmarsi.F', 'urascat.F', 'urascatm.F', 'urascor.F', 'zgroup.F',
    'dummy_mp.F', 'dummy_fdb.F', 'dummy_eclib.F', 'dummy_no_assimil.F',
    'dummy_no_nemo.F', 'decode_integrated_parameter.F',
    'decode_point_spectra.F', 'rfl4wam.F90', 'bouint.F', 'chief.F',
    'cpfdb.F', 'create_wam_bathymetry.F', 'intwaminput.F', 'preproc.F',
    'preset.F', 'write_currents.F', 'write_mpdecomp.F',
]
wam = Lib(name='wam', shared=False, source_dir=rootdir/'wam',
          pattern=pattern, ignore=wam_ignore + ignore)

mllibs = [ifsaux, algor, ecfftw, enkf, crm, ifs,
          radiation, satrad, surf, trans]  # wam, odb, bl, ifsobs, ifsiodummy

# Define final build targets
userlibs = []
support_libs = []
extlibs = []
oopsldlibs = []


fflags = ['-cpp', '-fno-second-underscore', '-fno-range-check', '-std=gnu']
omp = ['-fopenmp', '-m64', '-I/usr/local/apps/intel/16.0.3/compilers_and_libraries/linux/mkl/include']
fcopts = ['-O2', '-mavx']
fcfree = ['-ffree-form', '-ffree-line-length-0']
fcfixed = ['-ffixed-form']
autodbl_opts = ['-fdefault-real-8', '-fdefault-double-8']


class RapsGNUCompiler(GNUCompiler):
    """
    Defines the compiler flags and methods to use during compialtion.
    """
    F90 = str(bmdir/'external/gnu.melchior/install/bin/mpif90')
    F90FLAGS = ['-g'] + omp + ['-fbacktrace', '-fconvert=big-endian']
    F90FLAGS += defs + undefs + fcopts + fflags + fcfree

    FC = str(bmdir/'external/gnu.melchior/install/bin/mpif90')
    FCFLAGS = ['-g'] + omp + ['-fbacktrace', '-fconvert=big-endian']
    FCFLAGS += defs + undefs + fcopts + fflags + fcfixed

    CC = 'gcc'
    CFLAGS = ['-g', '-O2', '-mavx'] + defs

    LD = 'ar'
    LDFLAGS = ['-cr', '-g', '-fopenmp', '-fbacktrace', '-fconvert=big-endian',
               '-Wl,--as-needed' '-Wl,-export-dynamic', '-ffast-math']

    def link(self, objs, target, shared=True, logger=None, cwd=None):
        """
        Overriding the link function, so as to emulate RAPS linkage.
        """
        logger = logger or default_logger
        target = Path(target)
        if target.exists():
            delete(target)

        args = ['ar', '-cr', str(target)] + [str(o) for o in objs]
        execute(args, cwd=cwd)

        ranlib = ['ranlib', str(target)]
        execute(ranlib, cwd=cwd)


class WAMCompiler(RapsGNUCompiler):
    """
    Hacky way to override compiler flags (double-precision defautls)
    for the WAM lib.
    """

    # We add the autodbl options at the end to force default doubles
    F90FLAGS = ['-g'] + omp + ['-fbacktrace', '-fconvert=big-endian', '-c']
    F90FLAGS += defs + undefs + fcopts + fflags + fcfree + autodbl_opts

    FCFLAGS = ['-g'] + omp + ['-fbacktrace', '-fconvert=big-endian', '-c']
    FCFLAGS += defs + undefs + fcopts + fflags + fcfixed + autodbl_opts


###############################################################
#  Define the command line interfaces for the build script    #
###############################################################

@click.group()
def cli():
    pass


@cli.command('ifs')
@click.option('--compiler', '-cc', default='gnu',
              type=click.Choice(['gnu', 'intel-lxg', 'cray', 'custom']),
              help='Compiler toolchain to use for building IFS')
@click.option('-j', '--workers', default=None, type=int,
              help='Number of worker threads to use')
@click.option('-f', '--force', default=False, is_flag=True,
              help='Force a compleete re-build from scratch.')
def build_ifs(compiler, workers, force):
    """
    Builds the IFS in (almost) all it's glory.
    """
    build_dir = (bmdir/'build').relative_to(Path.cwd())
    include_dirs = [str(rootdir/i) for i in incdirs]
    include_dirs += [intfbroot, build_dir]

    # Initialize a parallel file logger that catches debug messages
    logger = FileLogger(name='loki-build', level=INFO, file_level=DEBUG,
                        filename=build_dir/'build.log', mode='a')
    compiler = RapsGNUCompiler()
    builder = Builder(compiler=compiler, workers=workers, logger=logger,
                      build_dir=build_dir, source_dirs=source_dirs,
                      include_dirs=include_dirs)

    if force:
        # Clean build directory, if we're forcing things.
        clean(build_dir, pattern=['**/*.o', '**/*.mod', '**/*.a', '**/*.log'])

    # Build the core component libs
    for lib in mllibs:
        lib.build(builder=builder)

    # TODO: This still creates erroneous results!
    # wam.build(builder=builder, compiler=WAMCompiler())

    # Somewhat crudely build up the final link command by emulating RAPS-17!
    # TODO: The final link is still very manual, since we still need a
    # few libs from the previous RAPS build.
    target = build_dir.absolute()/'ifsMASTER.dp.x'

    link_cmd = [compiler.f90]
    link_cmd += ['-g', '-fopenmp', '-m64',
                 '-I/usr/local/apps/intel/16.0.3/compilers_and_libraries/linux/mkl/include',
                 '-fbacktrace', '-fconvert=big-endian']
    link_cmd += ['-Wl,--as-needed', '-Wl,-export-dynamic', '-ffast-math',
                 '-Wl,-Map,/var/tmp/tmpdir/naml/jtmp.1694/tmpdir.gnu.melchior.dp/ifsload.map.V36_ORCA']

    # Define core objects and output binary
    link_cmd += ['/tmp/naml/ifs-build-dev/ifs/programs/master.o',
                 '/tmp/naml/ifs-build-dev/flexbuild/blacklist/C_code.o',
                 '-o', str(target),
                 '/var/tmp/tmpdir/naml/jtmp.1694/tmpdir.gnu.melchior.dp/dummies.o']

    # Link IFS libs (currently partially done from existing RAPS libs)
    link_cmd += ['-Wl,-rpath=%s' % (bmdir/'external/gnu.melchior/install/lib64'),
                 '-Wl,-rpath=%s' % (bmdir/'external/gnu.melchior/install/lib')]
    # TODO: Get everything built by loki-build, so that we can remove the top line.
    link_cmd += ['-L/tmp/naml/ifs-build-dev/flexbuild/build',  # Prefer our versions of the libs...
                 '-L/tmp/naml/ifs-build-dev/flexbuild',  # TODO: This brings in the leftovers; REMOVE!
                 '-Wl,--start-group', '-lalgor', '-lbl', '-lecfftw', '-lenkf', '-lifsobs', '-lcrm',
                 '-lifs', '-lifsaux', '-lodb', '-lradiation', '-lsatrad', '-lsurf', '-ltrans', '-lwam',
                 '-lifsiodummy', '-loopsifs_unboosted', '-lnemogribcoup.V36_ORCA', '-Wl,--end-group']

    # External linkages
    link_cmd += ['-L/tmp/naml/ifs-build-dev/flexbuild/external/gnu.melchior/install/lib64',
                 '-L/tmp/naml/ifs-build-dev/flexbuild/external/gnu.melchior/install/lib',
                 '-Wl,--no-as-needed', '-latlas_f', '-latlas', '-lfckit', '-leccodes_f90', '-leccodes', '-lpthread',
                 '-lmultio-fdb5', '-lmultio', '-lemosR64', '-lgfortran', '-lstdc++', '-lparmetis']
    link_cmd += ['-Wl,-rpath=/usr/local/apps/intel/16.0.3/compilers_and_libraries/linux/mkl/lib/intel64',
                 '-L/usr/local/apps/intel/16.0.3/compilers_and_libraries/linux/mkl/lib/intel64',
                 '-Wl,--no-as-needed', '-lmkl_gf_lp64', '-lmkl_sequential', '-lmkl_core', '-lpthread', '-lm', '-ldl',
                 '-lnetcdff', '-lnetcdf', '-lhdf5_hl', '-lhdf5hl_fortran', '-lhdf5', '-lz', '-lcurl',
                 '-lm', '-lrt', '-ldl']

    execute(link_cmd)


@cli.command('clean')
def build_clean():
    """
    Clean the local build directory.
    """

    build_dir = (bmdir/'build').relative_to(Path.cwd())
    clean(build_dir, pattern=['**/*.o', '**/*.mod', '**/*.a', '**/*.log'])


if __name__ == "__main__":
    cli()
