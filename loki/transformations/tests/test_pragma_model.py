# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Module, Subroutine, as_tuple
from loki.frontend import available_frontends

from loki.transformations import PragmaModelTransformation
from loki.ir import FindNodes, Pragma

def check_pragma(pragma, keyword, content, check_for_equality=True):
    assert pragma.keyword == keyword
    if check_for_equality:
        assert pragma.content == content
    else:
        for _content in as_tuple(content):
            assert _content in pragma.content

@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('directive', [False, 'openacc', 'omp-gpu'])
@pytest.mark.parametrize('keep_loki_pragmas', [True, False])
def test_transform_pragma_model(tmp_path, frontend, directive, keep_loki_pragmas):
    """
    Test Pragma model trafo for different directives/flavors.
    """
    fcode_mod = """
    module some_mod
      integer :: a, b
      !$loki create device(a, b)
    end module some_mod
    """.strip()

    fcode = """
subroutine some_func(ret)
  implicit none
  integer, intent(out) :: ret
  integer :: tmp1, tmp2, tmp3, tmp4, jk

  !$loki create device(tmp1, tmp2)
  !$loki update device(tmp1) host(tmp2)
  !$loki unstructured-data in(tmp1, tmp2) create(tmp3, tmp4) attach(tmp1)
  !$loki exit-unstructured-data out(tmp2, tmp3, tmp4) detach(tmp1) delete(tmp1) finalize
  !$loki structured-data in(tmp1) out(tmp2) inout(tmp3) create(tmp4)
  !$loki end structured-data in(tmp1) out(tmp2) inout(tmp3) create(tmp4)
  !$loki loop gang private(tmp1) vlength(128)
  !$loki end loop gang
  !$loki loop vector private(tmp2)
  !$loki end loop vector
  !$loki loop seq
  !$loki end loop seq
  !$loki routine vector
  !$loki routine seq
  !$loki device-present vars(tmp1, tmp2)
  !$loki end device-present vars(tmp1, tmp2)
  !$loki device-ptr vars(tmp1, tmp2)
  !$loki end device-ptr vars(tmp1, tmp2)
  !$loki unmapped-directive whatever(tmp1) foo(tmp2)
  ! misspelled by purpose
  !$loki create drvice(tmp1)
  !$loki structured-data present(tmp3, tmp4)
  !$loki end structured-data
  !$loki structured-data in(tmp1) present(tmp3, tmp4)
  !$loki end structured-data

end subroutine some_func
    """.strip()

    routine = Subroutine.from_source(fcode, frontend=frontend, xmods=[tmp_path])
    module = Module.from_source(fcode_mod, frontend=frontend, xmods=[tmp_path])

    pragma_model_trafo = PragmaModelTransformation(directive=directive,
            keep_loki_pragmas=keep_loki_pragmas)
    pragma_model_trafo.transform_subroutine(routine)
    pragma_model_trafo.transform_module(module)

    # CHECK MODULE
    pragmas = FindNodes(Pragma).visit(module.spec)
    if directive == 'openacc':
        check_pragma(pragmas[0], 'acc', 'declare create(a, b)')
    if directive == 'omp-gpu':
        check_pragma(pragmas[0], 'omp', 'declare target(a, b)')
    if directive is False and keep_loki_pragmas:
        check_pragma(pragmas[0], 'loki', 'create device(a, b)')

    # CHECK ROUTINE
    pragmas = FindNodes(Pragma).visit(routine.ir)
    if directive == 'openacc':
        args = (('acc', 'declare create(tmp1, tmp2)'),
                ('acc', ('update', 'device(tmp1)', 'self(tmp2)'), False),
                ('acc', ('enter data', 'copyin(tmp1, tmp2)', 'create(tmp3, tmp4)', 'attach(tmp1)'), False),
                ('acc', ('exit data', 'copyout(tmp2, tmp3, tmp4)', 'detach(tmp1)', 'delete(tmp1)', 'finalize'), False),
                ('acc', ('data', 'copyin(tmp1)', 'copy(tmp3)', 'copyout(tmp2)', 'create(tmp4)'), False),
                ('acc', 'end data'),
                ('acc', ('parallel loop gang', 'private(tmp1)', 'vector_length(128)'), False),
                ('acc', 'end parallel loop'),
                ('acc', ('loop vector', 'private(tmp2)'), False),
                ('loki', 'end loop vector'),
                ('acc', 'loop seq'),
                ('loki', 'end loop seq'),
                ('acc', 'routine vector'),
                ('acc', 'routine seq'),
                ('acc', 'data present(tmp1, tmp2)'),
                ('acc', 'end data'),
                ('acc', 'data deviceptr(tmp1, tmp2)'),
                ('acc', 'end data'),
                ('loki', 'unmapped-directive whatever(tmp1) foo(tmp2)'),
                ('loki', 'create drvice(tmp1)'),
                ('acc', 'data present(tmp3, tmp4)'),
                ('acc', 'end data'),
                ('acc', ('data', 'copyin(tmp1)', 'present(tmp3, tmp4)'), False),
                ('acc', 'end data'))
    if directive == 'omp-gpu':
        args = (('omp', 'declare target(tmp1, tmp2)'),
                ('omp', ('target update', 'to(tmp1)', 'from(tmp2)'), False),
                ('omp', ('target enter data', 'map(to: tmp1, tmp2)', 'map(alloc: tmp3, tmp4)'), False),
                ('omp', ('target exit data', 'map(from: tmp2, tmp3, tmp4)', 'map(delete: tmp1)'), False),
                ('omp', ('target data', 'map(to: tmp1)', 'map(tofrom: tmp3)',
                    'map(from: tmp2)', 'map(alloc: tmp4)'), False),
                ('omp', 'end target data'),
                ('omp', ('target teams distribute', 'thread_limit(128)'), False),
                ('omp', 'end target teams distribute'),
                ('omp', 'parallel do'),
                ('omp', 'end parallel do'),
                ('loki', 'loop seq'),
                ('loki', 'end loop seq'),
                ('loki', 'routine vector'),
                ('omp', 'declare target'),
                ('loki', ('device-present', 'vars(tmp1, tmp2)'), False),
                ('loki', ('end device-present', 'vars(tmp1, tmp2)'), False),
                ('loki', ('device-ptr', 'vars(tmp1, tmp2)'), False),
                ('loki', ('end device-ptr', 'vars(tmp1, tmp2)'), False),
                ('loki', 'unmapped-directive whatever(tmp1) foo(tmp2)'),
                ('loki', 'create drvice(tmp1)'),
                ('omp', 'target data map(to: tmp3, tmp4)'),
                ('omp', 'end target data'),
                ('omp', ('target data', 'map(to: tmp1, tmp3, tmp4)'), False),
                ('omp', 'end target data'))
    if directive is False:
        args = (('loki', 'create device(tmp1, tmp2)'),
                ('loki', ('update', 'device(tmp1)', 'host(tmp2)'), False),
                ('loki', ('unstructured-data', 'in(tmp1, tmp2)', 'create(tmp3, tmp4)', 'attach(tmp1)', ), False),
                ('loki', ('exit-unstructured-data', 'out(tmp2, tmp3, tmp4)', 'detach(tmp1)', 'delete(tmp1)',
                          'finalize'), False),
                ('loki', ('structured-data', 'in(tmp1)', 'out(tmp2)', 'inout(tmp3)', 'create(tmp4)'), False),
                ('loki', ('end structured-data', 'in(tmp1)', 'out(tmp2)', 'inout(tmp3)', 'create(tmp4)'), False),
                ('loki', ('loop gang', 'private(tmp1)', 'vlength(128)'), False),
                ('loki', 'end loop gang'),
                ('loki', ('loop vector', 'private(tmp2)'), False),
                ('loki', 'end loop vector'),
                ('loki', 'loop seq'),
                ('loki', 'end loop seq'),
                ('loki', 'routine vector'),
                ('loki', 'routine seq'),
                ('loki', ('device-present', 'vars(tmp1, tmp2)'), False),
                ('loki', ('end device-present', 'vars(tmp1, tmp2)'), False),
                ('loki', ('device-ptr', 'vars(tmp1, tmp2)'), False),
                ('loki', ('end device-ptr', 'vars(tmp1, tmp2)'), False),
                ('loki', 'unmapped-directive whatever(tmp1) foo(tmp2)'),
                ('loki', 'create drvice(tmp1)'),
                ('loki', 'structured-data present(tmp3, tmp4)'),
                ('loki', 'end structured-data'),
                ('loki', ('structured-data', 'in(tmp1)', 'present(tmp3, tmp4)'), False),
                ('loki', 'end structured-data'))

    if not keep_loki_pragmas:
        args = tuple(arg for arg in args if arg[0] != 'loki')

    for pragma, _args in zip(pragmas, args):
        check_pragma(pragma, *_args)
