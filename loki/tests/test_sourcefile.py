# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import pytest
import numpy as np

from loki import (
    Sourcefile, FindNodes, PreprocessorDirective, Intrinsic,
    Assignment, Import, fgen, ProcedureType, ProcedureSymbol,
    StatementFunction, Comment
)
from loki.jit_build import jit_compile, clean_test
from loki.frontend import available_frontends, OMNI


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI, 'Files are preprocessed')]))
def test_sourcefile_pp_macros(here, frontend):
    filepath = here/'sources/sourcefile_pp_macros.F90'
    routine = Sourcefile.from_file(filepath, frontend=frontend)['routine_pp_macros']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)
    assert len(directives) == 8
    assert all(node.text.startswith('#') for node in directives)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[
    (OMNI, 'Files are preprocessed')
]))
def test_sourcefile_pp_directives(here, frontend):
    filepath = here/'sources/sourcefile_pp_directives.F90'
    routine = Sourcefile.from_file(filepath, frontend=frontend)['routine_pp_directives']

    # Note: these checks are rather loose as we currently do not restore the original version but
    # simply replace the PP constants by strings
    directives = FindNodes(PreprocessorDirective).visit(routine.body)
    assert len(directives) == 1
    assert directives[0].text == '#define __FILENAME__ __FILE__'
    intrinsics = FindNodes(Intrinsic).visit(routine.body)
    assert '__FILENAME__' in intrinsics[0].text and '__DATE__' in intrinsics[0].text
    assert '__FILE__' in intrinsics[1].text and '__VERSION__' in intrinsics[1].text

    statements = FindNodes(Assignment).visit(routine.body)
    assert len(statements) == 1
    assert fgen(statements[0]) == 'y = 0*5 + 0'


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_pp_include(here, frontend):
    filepath = here/'sources/sourcefile_pp_include.F90'
    sourcefile = Sourcefile.from_file(filepath, frontend=frontend, includes=[here/'include'])
    routine = sourcefile['routine_pp_include']

    statements = FindNodes(Assignment).visit(routine.body)
    assert len(statements) == 1
    if frontend == OMNI:
        # OMNI resolves that statement function!
        assert fgen(statements[0]) == 'c = real(a + b, kind=4)'
    else:
        assert fgen(statements[0]) == 'c = add(a, b)'

    if frontend is not OMNI:
        # OMNI resolves the import in the frontend
        imports = FindNodes(Import).visit([routine.spec, routine.body])
        assert len(imports) == 1
        assert imports[0].c_import
        assert imports[0].module == 'some_header.h'


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_cpp_preprocessing(here, frontend):
    """
    Test the use of the external CPP-preprocessor.
    """
    filepath = here/'sources/sourcefile_cpp_preprocessing.F90'

    source = Sourcefile.from_file(filepath, preprocess=True, frontend=frontend)
    routine = source['sourcefile_external_preprocessing']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)

    if frontend is not OMNI:
        # OMNI skips the import in the frontend
        imports = FindNodes(Import).visit([routine.spec, routine.body])
        assert len(imports) == 1
        assert imports[0].c_import
        assert imports[0].module == 'some_header.h'

    assert len(directives) == 0
    assert 'b = 123' in fgen(routine)

    # Check that the ``define`` gets propagated correctly
    source = Sourcefile.from_file(filepath, preprocess=True, defines='FLAG_SMALL',
                                  frontend=frontend)
    routine = source['sourcefile_external_preprocessing']
    directives = FindNodes(PreprocessorDirective).visit(routine.ir)

    assert len(directives) == 0
    assert 'b = 6' in fgen(routine)


@pytest.mark.parametrize('frontend', available_frontends())
def test_sourcefile_cpp_stmt_func(here, frontend, tmp_path):
    """
    Test the correct identification of statement functions
    after inlining by preprocessor.
    """
    sourcepath = here/'sources'
    filepath = sourcepath/'sourcefile_cpp_stmt_func.F90'

    source = Sourcefile.from_file(filepath, includes=sourcepath, preprocess=True, frontend=frontend, xmods=[tmp_path])
    module = source['cpp_stmt_func_mod']
    module.name += f'_{frontend!s}'

    # OMNI inlines statement functions, so we can't check the representation
    if frontend != OMNI:
        routine = source['cpp_stmt_func']
        stmt_func_decls = FindNodes(StatementFunction).visit(routine.spec)
        assert len(stmt_func_decls) == 4

        for decl in stmt_func_decls:
            var = routine.variable_map[str(decl.variable)]
            assert isinstance(var, ProcedureSymbol)
            assert isinstance(var.type.dtype, ProcedureType)
            assert var.type.dtype.procedure is decl
            assert decl.source is not None

    # Generate code and compile
    filepath = tmp_path/f'{module.name}.f90'
    mod = jit_compile(source, filepath=filepath, objname=module.name)

    # Verify it produces correct results
    klon, klev = 10, 5
    kidia, kfdia = 1, klon
    zfoeew = np.zeros((klon, klev), order='F')
    mod.cpp_stmt_func(kidia, kfdia, klon, klev, zfoeew)
    assert (zfoeew == 0.25).all()

    clean_test(filepath)




@pytest.mark.parametrize('frontend', available_frontends(include_regex=True))
def test_sourcefile_clone(frontend, tmp_path):
    """
    Make sure cloning a source file works as expected
    """
    fcode = """
! Comment outside
module my_mod
  implicit none
  contains
    subroutine my_routine
      implicit none
    end subroutine my_routine
end module my_mod

subroutine other_routine
  use my_mod, only: my_routine
  implicit none
  call my_routine()
end subroutine other_routine
    """.strip()
    source = Sourcefile.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Clone the source file twice
    new_source = source.clone()
    new_new_source = source.clone()

    # Apply some changes that should only be affecting each clone
    new_source['other_routine'].name = 'new_name'
    new_new_source['my_mod']['my_routine'].name = 'new_mod_routine'

    assert 'other_routine' in source
    assert 'other_routine' not in new_source
    assert 'other_routine' in new_new_source

    assert 'new_name' not in source
    assert 'new_name' in new_source
    assert 'new_name' not in new_new_source

    assert 'my_mod' in source
    assert 'my_mod' in new_source
    assert 'my_mod' in new_new_source

    assert 'my_routine' in source['my_mod']
    assert 'my_routine' in new_source['my_mod']
    assert 'my_routine' not in new_new_source['my_mod']

    assert 'new_mod_routine' not in source['my_mod']
    assert 'new_mod_routine' not in new_source['my_mod']
    assert 'new_mod_routine' in new_new_source['my_mod']

    if not source._incomplete:
        assert isinstance(source.ir.body[0], Comment)
        comment_text = source.ir.body[0].text
        new_comment_text = comment_text + ' some more text'
        source.ir.body[0]._update(text=new_comment_text)

        assert source.ir.body[0].text == new_comment_text
        assert new_source.ir.body[0].text == comment_text
        assert new_new_source.ir.body[0].text == comment_text
    else:
        assert new_source._incomplete
        assert new_new_source._incomplete

        assert source['other_routine']._incomplete
        assert new_source['new_name']._incomplete
        assert new_new_source['other_routine']._incomplete

        assert new_source['new_name']._parser_classes == source['other_routine']._parser_classes
        assert new_new_source['other_routine']._parser_classes == source['other_routine']._parser_classes

        mod = source['my_mod']
        new_mod = new_source['my_mod']
        new_new_mod = new_new_source['my_mod']

        assert mod._incomplete
        assert new_mod._incomplete
        assert new_new_mod._incomplete

        assert new_mod._parser_classes == mod._parser_classes
        assert new_new_mod._parser_classes == mod._parser_classes

        assert mod['my_routine']._incomplete
        assert new_mod['my_routine']._incomplete
        assert new_new_mod['new_mod_routine']._incomplete

        assert new_mod['my_routine']._parser_classes == mod['my_routine']._parser_classes
        assert new_new_mod['new_mod_routine']._parser_classes == mod['my_routine']._parser_classes
