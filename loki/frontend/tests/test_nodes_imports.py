# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Verify correct frontend behaviour for import and access-spec parsing.
"""

import pytest

from loki import Module, fgen
from loki.frontend import available_frontends, OMNI
from loki.ir import nodes as ir, FindNodes
from loki.jit_build import jit_compile_lib


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_access_spec_none(frontend, tmp_path):
    """
    Test correct parsing without access-spec statements
    """
    fcode = """
module test_access_spec_mod
    implicit none

    integer pub_var = 1
contains
    subroutine routine
        integer i
        i = pub_var
    end subroutine routine
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Check module properties
    assert module.default_access_spec is None
    assert isinstance(module.public_access_spec, tuple) and not module.public_access_spec
    assert isinstance(module.private_access_spec, tuple) and not module.private_access_spec

    # Check backend output
    code = module.to_fortran().upper()
    assert 'PUBLIC' not in code
    assert 'PRIVATE' not in code

    # Check that property has not propagated to symbol type
    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is None
    assert pub_var.type.private is None

    # Check properties after clone
    new_module = module.clone(
        default_access_spec='PUBLIC', public_access_spec='PUB_VAR',
        private_access_spec='ROUTINE'
    )
    assert new_module.default_access_spec == 'public'
    assert new_module.public_access_spec == ('pub_var',)
    assert new_module.private_access_spec == ('routine',)


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Inlines access-spec as declaration attr')]))
def test_module_access_spec_private(frontend, tmp_path):
    """
    Test correct parsing of access-spec statements with default private
    """
    fcode = """
module test_access_spec_mod
    implicit none
    private
    public :: pub_var, routine
    PRIVATE OTHER_PRIVATE_VAR

    integer pub_var = 1
    integer private_var = 2
    integer other_private_var = 3
contains
    subroutine routine
        integer i
        i = pub_var
    end subroutine routine
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Check module properties
    assert module.default_access_spec == 'private'
    assert module.public_access_spec == ('pub_var', 'routine')
    assert module.private_access_spec == ('other_private_var',)

    # Check backend output
    code = module.to_fortran().upper()
    assert 'PUBLIC\n' not in code
    assert 'PUBLIC :: PUB_VAR, ROUTINE' in code
    assert 'PRIVATE\n' in code
    assert 'PRIVATE :: OTHER_PRIVATE_VAR' in code

    # Check that property has not propagated to symbol type
    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is None
    assert pub_var.type.private is None

    # Check properties after clone
    new_module = module.clone(private_access_spec=None)
    assert new_module.default_access_spec == 'private'
    assert new_module.public_access_spec == ('pub_var', 'routine')
    assert new_module.private_access_spec == ()


@pytest.mark.parametrize('frontend', available_frontends(skip=[(OMNI, 'Inlines access-spec as declaration attr')]))
def test_module_access_spec_public(frontend, tmp_path):
    """
    Test correct parsing of access-spec statements with default public
    """
    fcode = """
module test_access_spec_mod
    implicit none
    PUBLIC
    PUBLIC ROUTINE
    private :: private_var, other_private_var

    integer pub_var = 1
    integer private_var = 2
    integer other_private_var = 3
contains
    subroutine routine
        integer i
        i = pub_var
    end subroutine routine
end module test_access_spec_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Check module properties
    assert module.default_access_spec == 'public'
    assert module.public_access_spec == ('routine', )
    assert module.private_access_spec == ('private_var', 'other_private_var')

    # Check backend output
    code = module.to_fortran().upper()
    assert 'PUBLIC\n' in code
    assert 'PUBLIC :: ROUTINE' in code
    assert 'PRIVATE\n' not in code
    assert 'PRIVATE :: PRIVATE_VAR, OTHER_PRIVATE_VAR' in code

    # Check that property has not propagated to symbol type
    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is None
    assert pub_var.type.private is None

    # Check properties after clone
    new_module = module.clone(
        default_access_spec='PRivate', public_access_spec=('ROUTINE', 'pub_var')
    )
    assert new_module.default_access_spec == 'private'
    assert new_module.public_access_spec == ('routine', 'pub_var')
    assert new_module.private_access_spec == ('private_var', 'other_private_var')


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_access_attr(frontend, tmp_path):
    """
    Test correct parsing of access-spec attributes
    """
    fcode = """
module test_access_attr_mod
    implicit none
    private
    integer, public :: pub_var
    integer :: unspecified_var
    integer, private :: priv_var
    integer :: other_var
end module test_access_attr_mod
    """.strip()

    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    code = module.to_fortran().upper()

    priv_var = module.variable_map['priv_var']
    assert priv_var.type.private is True
    assert priv_var.type.public is None

    pub_var = module.variable_map['pub_var']
    assert pub_var.type.public is True
    assert pub_var.type.private is None

    unspecified_var = module.variable_map['unspecified_var']
    other_var = module.variable_map['other_var']

    assert unspecified_var.type.public is None
    assert other_var.type.public is None

    if frontend == OMNI:  # OMNI applies access spec to each variable
        assert code.count('PRIVATE') == 3
        assert unspecified_var.type.private is True
        assert other_var.type.private is True
    else:
        assert code.count('PRIVATE') == 2
        assert unspecified_var.type.private is None
        assert other_var.type.private is None
    assert code.count('PUBLIC') == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_rename_imports_with_definitions(frontend, tmp_path):
    """
    Test use statement with rename lists
    """
    fcode_mod1 = """
module test_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_rename_mod
    """.strip()

    fcode_mod2 = """
module test_other_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_other_rename_mod
    """.strip()

    fcode_mod3 = """
module some_mod
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end module some_mod
    """.strip()

    mod1 = Module.from_source(fcode_mod1, frontend=frontend, xmods=[tmp_path])
    mod2 = Module.from_source(fcode_mod2, frontend=frontend, xmods=[tmp_path])
    mod3 = Module.from_source(fcode_mod3, frontend=frontend, xmods=[tmp_path], definitions=[mod1, mod2])

    # Check all entries exist in the symbol table
    mod1_imports = {
        'first_var1': 'var1',
        'var2': None,
        'first_var3': 'var3'
    }
    mod2_imports = {
        'second_var1': 'var1',
        'other_var2': 'var2',
        'other_var3': 'var3'
    }
    expected_symbols = list(mod1_imports) + list(mod2_imports)
    for s in expected_symbols:
        assert s in mod3.symbol_attrs

    # Check that var1 has note been imported under that name
    assert 'var1' not in mod3.symbol_attrs

    # Verify correct symbol attributes
    for s, use_name in mod1_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is mod1
        assert mod3.symbol_attrs[s].use_name == use_name
        assert mod3.symbol_attrs[s].compare(mod1.symbol_attrs[use_name or s], ignore=('imported', 'module', 'use_name'))
    for s, use_name in mod2_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is mod2
        assert mod3.symbol_attrs[s].use_name == use_name
        assert mod3.symbol_attrs[s].compare(mod2.symbol_attrs[use_name or s], ignore=('imported', 'module', 'use_name'))

    # Verify Import IR node
    for imprt in FindNodes(ir.Import).visit(mod3.spec):
        if imprt.module == 'test_rename_mod':
            assert imprt.rename_list
            assert not imprt.symbols
            assert 'var1' in dict(imprt.rename_list)
            assert 'var3' in dict(imprt.rename_list)
        else:
            assert not imprt.rename_list
            assert imprt.symbols

    # Verify fgen output
    fcode = fgen(mod3)
    for s, use_name in mod1_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode
    for s, use_name in mod2_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_rename_imports_no_definitions(frontend, tmp_path):
    """
    Test use statement with rename lists when definitions are not available
    """
    fcode_mod1 = """
module test_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_rename_mod
    """.strip()

    fcode_mod2 = """
module test_other_rename_mod
    implicit none
    integer :: var1
    integer :: var2
    integer :: var3
end module test_other_rename_mod
    """.strip()

    _ = Module.from_source(fcode_mod1, frontend=frontend, xmods=[tmp_path])
    _ = Module.from_source(fcode_mod2, frontend=frontend, xmods=[tmp_path])

    fcode_mod3 = """
module some_mod
    use test_rename_mod, first_var1 => var1, first_var3 => var3
    use test_other_rename_mod, only: second_var1 => var1
    use test_other_rename_mod, only: other_var2 => var2, other_var3 => var3
    implicit none
end module some_mod
    """.strip()

    mod3 = Module.from_source(fcode_mod3, frontend=frontend, xmods=[tmp_path])

    # Check all entries exist in the symbol table
    mod1_imports = {
        'first_var1': 'var1',
        'first_var3': 'var3'
    }
    mod2_imports = {
        'second_var1': 'var1',
        'other_var2': 'var2',
        'other_var3': 'var3'
    }
    expected_symbols = list(mod1_imports) + list(mod2_imports)
    for s in expected_symbols:
        assert s in mod3.symbol_attrs

    # Check that var1 has note been imported under that name
    assert 'var1' not in mod3.symbol_attrs
    assert 'var2' not in mod3.symbol_attrs

    # Verify correct symbol attributes
    for s, use_name in mod1_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is None
        assert mod3.symbol_attrs[s].use_name == use_name
    for s, use_name in mod2_imports.items():
        assert mod3.symbol_attrs[s].imported
        assert mod3.symbol_attrs[s].module is None
        assert mod3.symbol_attrs[s].use_name == use_name

    # Verify Import IR node
    for imprt in FindNodes(ir.Import).visit(mod3.spec):
        if imprt.module == 'test_rename_mod':
            assert imprt.rename_list
            assert not imprt.symbols
            assert 'var1' in dict(imprt.rename_list)
            assert 'var3' in dict(imprt.rename_list)
        else:
            assert not imprt.rename_list
            assert imprt.symbols

    # Verify fgen output
    fcode = fgen(mod3)
    for s, use_name in mod1_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode
    for s, use_name in mod2_imports.items():
        assert use_name is None or f'{s} => {use_name}' in fcode


@pytest.mark.parametrize('frontend', available_frontends())
def test_module_use_module_nature(frontend, tmp_path):
    """
    Test module natures attributes in ``USE`` statements
    """
    mcode = """
module iso_fortran_env
    use, intrinsic :: iso_c_binding, only: int16 => c_int16_t
    implicit none
    integer, parameter :: int8 = int16
end module iso_fortran_env
    """.strip()

    fcode = """
module module_nature_mod
    implicit none
contains
    subroutine inquire_my_kinds(i8, i16)
        use, non_intrinsic :: iso_fortran_env, only: int8, int16
        integer, intent(out) :: i8, i16
        i8 = int8
        i16 = int16
    end subroutine inquire_my_kinds
    subroutine inquire_kinds(i8, i16)
        use, intrinsic :: iso_fortran_env, only: int8, int16
        integer, intent(out) :: i8, i16
        i8 = int8
        i16 = int16
    end subroutine inquire_kinds
end module module_nature_mod
    """.strip()

    ext_mod = Module.from_source(mcode, frontend=frontend, xmods=[tmp_path])

    # Check properties on the Import IR node in the external module
    assert ext_mod.imported_symbols == ('int16',)
    imprt = FindNodes(ir.Import).visit(ext_mod.spec)[0]
    assert imprt.nature.lower() == 'intrinsic'
    assert imprt.module.lower() == 'iso_c_binding'
    assert ext_mod.imported_symbol_map['int16'].type.imported is True
    assert ext_mod.imported_symbol_map['int16'].type.module is None

    if frontend == OMNI:
        # OMNI throws Syntax Error on NON_INTRINSIC...
        fcode = fcode.replace('use, non_intrinsic ::', 'use')

    mod = Module.from_source(fcode, frontend=frontend, definitions=[ext_mod], xmods=[tmp_path])

    # Check properties on the Import IR node in both routines
    my_kinds = mod['inquire_my_kinds']
    kinds = mod['inquire_kinds']

    assert set(my_kinds.imported_symbols) == {'int8', 'int16'}
    assert set(kinds.imported_symbols) == {'int8', 'int16'}

    my_import_map = {s.name: imprt for imprt in FindNodes(ir.Import).visit(my_kinds.spec) for s in imprt.symbols}
    import_map = {s.name: imprt for imprt in FindNodes(ir.Import).visit(kinds.spec) for s in imprt.symbols}

    assert my_import_map['int8'] is my_import_map['int16']
    assert import_map['int8'] is import_map['int16']

    if frontend == OMNI:
        assert my_import_map['int8'].nature is None
    else:
        assert my_import_map['int8'].nature.lower() == 'non_intrinsic'
    assert my_import_map['int8'].module.lower() == 'iso_fortran_env'
    assert import_map['int8'].nature.lower() == 'intrinsic'
    assert import_map['int8'].module.lower() == 'iso_fortran_env'

    # Check type annotations for imported symbols
    assert all(s.type.imported is True for s in my_kinds.imported_symbols)
    assert all(s.type.imported is True for s in kinds.imported_symbols)

    assert my_kinds.imported_symbol_map['int8'].type.module is ext_mod
    assert my_kinds.imported_symbol_map['int16'].type.module is ext_mod

    assert kinds.imported_symbol_map['int8'].type.module is None
    assert kinds.imported_symbol_map['int16'].type.module is None

    # Sanity check fgen
    assert 'use, intrinsic' in ext_mod.to_fortran().lower()
    if frontend != OMNI:
        assert 'use, non_intrinsic' in my_kinds.to_fortran().lower()
    assert 'use, intrinsic' in kinds.to_fortran().lower()

    # Verify JIT compile
    lib = jit_compile_lib([ext_mod, mod], path=tmp_path, name=mod.name)
    my_kinds_func = lib.module_nature_mod.inquire_my_kinds
    kinds_func = lib.module_nature_mod.inquire_kinds

    my_i8, my_i16 = my_kinds_func()
    i8, i16 = kinds_func()

    assert my_i8 == my_i16
    assert i8 < i16
    assert my_i8 == i16
    assert my_i8 == lib.iso_fortran_env.int8


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('ext2_import', [
    ('use ext2, only: ext2_var1, ext2_var2', ('ext2_var1', 'ext2_var2')),
    ('use ext2', ()),
    ('use ext2, ext2_var => ext2_var1', ('ext2_var',)),
    ('use ext2, only: ext2_var2', ('ext2_var2',))
    ])
def test_subroutine_imported_symbols(tmp_path, frontend, ext2_import):
    """ Test return of imported symbols """
    fcode_ext1_mod = """
    module ext1
      implicit none
      integer :: ext1_var1, ext1_var2, ext1_var3
    end module ext1
    """

    fcode_ext2_mod = """
    module ext2
      implicit none
      integer :: ext2_var1, ext2_var2
    end module ext2
    """

    fcode_ext3_mod = """
    module ext3
      implicit none
      integer :: ext3_var1, ext3_var2
    end module ext3
    """

    fcode_module = f"""
module parent_mod
  use ext1, only: ext1_var1
  {ext2_import[0]}
  implicit none
  contains
   subroutine routine1(a)
      use ext1, only: ext1_var2, ext1_var3
      use ext3
      integer, intent(inout) :: a(:)
   end subroutine routine1

   subroutine routine2(b)
      use ext3, only: ext3_var1, ext3_var2
      integer, intent(inout) :: b(:)
   end subroutine routine2
end module parent_mod
    """

    ext1_mod = Module.from_source(fcode_ext1_mod, frontend=frontend, xmods=[tmp_path])
    ext2_mod = Module.from_source(fcode_ext2_mod, frontend=frontend, xmods=[tmp_path])
    ext3_mod = Module.from_source(fcode_ext3_mod, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(
        fcode_module, frontend=frontend, xmods=[tmp_path], definitions=[ext1_mod, ext2_mod, ext3_mod]
    )
    routine1 = module.subroutines[0]
    routine2 = module.subroutines[1]

    # get imported_symbols and all_imported_symbols
    mod_imp_symbols = set(module.imported_symbols)
    mod_all_imp_symbols = set(module.all_imported_symbols)
    routine1_imp_symbols = set(routine1.imported_symbols)
    routine1_all_imp_symbols = set(routine1.all_imported_symbols)
    routine2_imp_symbols = set(routine2.imported_symbols)
    routine2_all_imp_symbols = set(routine2.all_imported_symbols)

    # check/test results
    exp_mod_imp_symbols = set(('ext1_var1',) + ext2_import[1])
    assert mod_imp_symbols == exp_mod_imp_symbols
    for var in exp_mod_imp_symbols:
        assert var in module.all_imported_symbol_map
    exp_routine1_imp_symbols = set(['ext1_var2', 'ext1_var3'])
    assert routine1_imp_symbols == exp_routine1_imp_symbols
    exp_routine2_imp_symbols = set(['ext3_var1', 'ext3_var2'])
    assert routine2_imp_symbols == exp_routine2_imp_symbols
    assert mod_imp_symbols == mod_all_imp_symbols
    assert routine1_all_imp_symbols == exp_routine1_imp_symbols | exp_mod_imp_symbols
    for var in exp_routine1_imp_symbols | exp_mod_imp_symbols:
        assert var in routine1.all_imported_symbol_map
    assert routine2_all_imp_symbols == exp_routine2_imp_symbols | exp_mod_imp_symbols
    for var in exp_routine2_imp_symbols | exp_mod_imp_symbols:
        assert var in routine2.all_imported_symbol_map
