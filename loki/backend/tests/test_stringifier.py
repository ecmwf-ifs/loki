# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module
from loki.backend import DefaultStyle, Stringifier
from loki.frontend import available_frontends, OMNI


@pytest.mark.parametrize('frontend', available_frontends())
def test_stringifier(frontend, tmp_path):
    """
    Test basic stringifier capability for most IR nodes.
    """
    fcode = """
MODULE some_mod
  INTEGER :: n
  !$loki dimension(klon)
  REAL :: arr(:)
  CONTAINS
    SUBROUTINE some_routine (x, y)
      ! This is a basic subroutine with some loops
      IMPLICIT NONE
      REAL, INTENT(IN) :: x
      REAL, INTENT(OUT) :: y
      INTEGER :: i
      ! And now to the content
      IF (x < 1E-8 .and. x > -1E-8) THEN
        x = 0.
      ELSE IF (x > 0.) THEN
        DO WHILE (x > 1.)
          x = x / 2.
        ENDDO
      ELSE
        x = -x
      ENDIF
      y = 0
      DO i=1,n
        y = y + x*x
      ENDDO
      y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1.
    END SUBROUTINE some_routine
    FUNCTION my_sqrt (arg)
      IMPLICIT NONE
      REAL, INTENT(IN) :: arg
      REAL :: my_sqrt
      my_sqrt = SQRT(arg)
    END FUNCTION my_sqrt
  SUBROUTINE other_routine (m)
    ! This is just to have some more IR nodes
    ! with multi-line comments and everything...
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: m
    REAL, ALLOCATABLE :: var(:)
    !$loki some pragma
    SELECT CASE (m)
      CASE (0)
        m = 1
      CASE (1:10)
        PRINT *, '1 to 10'
      CASE (-1, -2)
        m = 10
      CASE DEFAULT
        PRINT *, 'Default case'
    END SELECT
    ASSOCIATE (x => arr(m))
      x = x * 2.
    END ASSOCIATE
    ALLOCATE(var, source=arr)
    CALL some_routine (arr(1), var(1))
    arr(:) = arr(:) + var(:)
    DEALLOCATE(var)
  END SUBROUTINE other_routine
END MODULE some_mod
    """.strip()
    ref_lines = [
        "<Module:: some_mod>",  # l. 1
        "#<Section::>",
        "##<VariableDeclaration:: n>",
        "##<Pragma:: loki dimension(klon)>",
        "##<VariableDeclaration:: arr(:)>",
        "#<Subroutine:: some_routine>",
        "##<Comment:: ! This is a b...>",
        "##<Section::>",
        "###<Intrinsic:: IMPLICIT NONE>",
        "###<VariableDeclaration:: x>",  # l. 10
        "###<VariableDeclaration:: y>",
        "###<VariableDeclaration:: i>",
        "##<Section::>",
        "###<Comment:: ! And now to ...>",
        "###<Conditional::>",
        "####<If x < 1E-8 and x > -1E-8>",
        "#####<Assignment:: x = 0.>",
        "####<Else>",
        "#####<Conditional::>",
        "######<If x > 0.>",  # l. 20
        "#######<WhileLoop:: x > 1.>",
        "########<Assignment:: x = x / 2.>",
        "######<Else>",
        "#######<Assignment:: x = -x>",
        "###<Assignment:: y = 0>",
        "###<Loop:: i=1:n>",
        "####<Assignment:: y = y + x*x>",
        "###<Assignment:: y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + ",
        "... 1. + 1.>",
        "#<Function:: my_sqrt>", # l. 30
        "##<Section::>",
        "###<Intrinsic:: IMPLICIT NONE>",
        "###<VariableDeclaration:: arg>",
        "###<VariableDeclaration:: my_sqrt>",
        "##<Section::>",
        "###<Assignment:: my_sqrt = SQRT(arg)>",
        "#<Subroutine:: other_routine>",
        "##<CommentBlock:: ! This is jus...>",
        "##<Section::>",
        "###<Intrinsic:: IMPLICIT NONE>",  # l. 40
        "###<VariableDeclaration:: m>",
        "###<VariableDeclaration:: var(:)>",
        "##<Section::>",
        "###<Pragma:: loki some pragma>",
        "###<MultiConditional:: m>",
        "####<Case (0)>",
        "#####<Assignment:: m = 1>",
        "####<Case (1:10)>",
        "#####<Intrinsic:: PRINT *, '1 t...>",
        "####<Case (-1, -2)>",  # l. 50
        "#####<Assignment:: m = 10>",
        "####<Default>",
        "#####<Intrinsic:: PRINT *, 'Def...>",
        "###<Associate:: arr(m)=x>",
        "####<Assignment:: x = x*2.>",
        "###<Allocation:: var>",
        "###<Call:: some_routine>",
        "###<Assignment:: arr(:) = arr(:) + var(:)>",
        "###<Deallocation:: var>",
    ]

    if frontend == OMNI:
        # Some string inconsistencies
        ref_lines[15] = ref_lines[15].replace('1E-8', '1e-8')
        ref_lines[35] = ref_lines[35].replace('SQRT', 'sqrt')
        ref_lines[48] = ref_lines[48].replace('PRINT', 'print')
        ref_lines[52] = ref_lines[52].replace('PRINT', 'print')

    cont_index = 27  # line number where line continuation is happening
    ref = '\n'.join(ref_lines)
    module = Module.from_source(fcode, frontend=frontend, xmods=[tmp_path])

    # Test custom indentation
    def line_cont(indent):
        return f'\n{"...":{max(len(indent), 1)}} '

    assert Stringifier(
        style=DefaultStyle(indent_char='#'), line_cont=line_cont
    ).visit(module).strip() == ref.strip()

    # Test default
    ref_lines = ref.strip().replace('#', '  ').splitlines()
    ref_lines[cont_index] = '      <Assignment:: y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1.'
    ref_lines[cont_index + 1] = '       + 1. + 1.>'
    default_ref = '\n'.join(ref_lines)
    assert Stringifier(style=DefaultStyle()).visit(module).strip() == default_ref

    # Test custom initial depth
    ref_lines = ['#' + line if line else '' for line in ref.splitlines()]
    ref_lines[cont_index] = '####<Assignment:: y = my_sqrt(y) + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. +'
    ref_lines[cont_index + 1] = '...   1. + 1.>'
    depth_ref = '\n'.join(ref_lines)
    assert Stringifier(
        style=DefaultStyle(indent_char='#'), depth=1, line_cont=line_cont
    ).visit(module).strip() == depth_ref

    # Test custom linewidth
    ref_lines = ref.strip().splitlines()
    ref_lines = ref_lines[:cont_index] + ['###<Assignment:: y = my_sqrt(y) + 1. + 1. +',
                                          '...  1. + 1. + 1. + 1. + 1. + 1. + 1. + 1. ',
                                          '... + 1. + 1. + 1.>'] + ref_lines[cont_index+2:]
    w_ref = '\n'.join(ref_lines)
    assert Stringifier(
        style=DefaultStyle(indent_char='#', linewidth=44), line_cont=line_cont
    ).visit(module).strip() == w_ref
