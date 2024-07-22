# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

from loki.batch import Pipeline, Transformation
from loki.transformations.hoist_variables import HoistTemporaryArraysAnalysis
from loki.transformations.single_column.base import SCCBaseTransformation
from loki.transformations.single_column.vector import (
    SCCDevectorTransformation, SCCRevectorTransformation, SCCDemoteTransformation
)
from loki.transformations.single_column.scc_cuf import (
    HoistTemporaryArraysDeviceAllocatableTransformation,
    HoistTemporaryArraysPragmaOffloadTransformation,
    SccLowLevelDataOffload, SccLowLevelLaunchConfiguration
)
from loki.transformations.block_index_transformations import (
        InjectBlockIndexTransformation,
        LowerBlockIndexTransformation, LowerBlockLoopTransformation,
        LowerConstantArrayIndex
)
from loki.transformations.transform_derived_types import DerivedTypeArgumentsTransformation
from loki.transformations.data_offload import (
    GlobalVariableAnalysis, GlobalVarHoistTransformation, GlobalVarOffloadTransformation
)
from loki.transformations.parametrise import ParametriseTransformation, ParametriseArrayDimsTransformation
from loki.transformations.inline import (
    inline_constant_parameters, inline_elemental_functions
)
from loki.transformations.argument_shape import (
    ExplicitArgumentArrayShapeTransformation, ArgumentArrayShapeAnalysis 
)

__all__ = [
        'SCCLowLevelCufHoist', 'SCCLowLevelCufParametrise', 'SCCLowLevelHoist',
        'SCCLowLevelParametrise', 'SCCLowLevelCuf'
]

def inline_elemental_kernel(routine, **kwargs):
    role = kwargs['role']

    if role == 'kernel':

        inline_constant_parameters(routine, external_only=True)
        inline_elemental_functions(routine)


class InlineTransformation(Transformation):

    def transform_subroutine(self, routine, **kwargs):
        role = kwargs['role']

        if role == 'kernel':

            SCCBaseTransformation.explicit_dimensions(routine)
            # inline_constant_parameters(routine, external_only=True)
            inline_elemental_functions(routine)


"""
The basic Single Column Coalesced low-level GPU via CUDA-Fortran (SCC-CUF).

This tranformation will convert kernels with innermost vectorisation
along a common horizontal dimension to a GPU-friendly loop-layout via
loop inversion and local array variable demotion. The resulting kernel
remains "vector-parallel", but with the ``horizontal`` loop as the
outermost iteration dimension (as far as data dependencies
allow). This allows local temporary arrays to be demoted to scalars,
where possible.

Kernels are specified via ``'GLOBAL'`` and the number of threads that
execute the kernel for a given call is specified via the chevron syntax.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:
1. :any:`SCCBaseTransformation` - Ensure utility variables and resolve
   problematic code constructs.
2. :any:`SCCDevectorTransformation` - Remove horizontal vector loops.
3. :any:`SCCDemoteTransformation` - Demote local temporary array
   variables where appropriate.
4. :any:`SCCRevectorTransformation` - Re-insert the vecotr loops outermost,
   according to identified vector sections.
5. :any:`LowerBlockIndexTransformation` - Lower the block index (for
   array argument definitions).
6. :any:`InjectBlockIndexTransformation` - Complete the previous step
   and inject the block index for the relevant arrays.
7. :any:`LowerBlockLoopTransformation` - Lower the block loop
   from driver to kernel(s).
8. :any:`SCCLowLevelLaunchConfiguration` - Create launch configuration
   and related things.
9. :any:`SCCLowLevelDataOffload` - Create/handle data offload
   and related things.

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the horizontal data dimension and iteration space.
block_dim : :any:`Dimension`
    Optional ``Dimension`` object to define the blocking dimension
    to use for hoisted column arrays if hoisting is enabled.
directive : string or None
    Directives flavour to use for parallelism annotations; either
    ``'openacc'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
derived_types: tuple
    List of relevant derived types
transformation_type : str
    Kind of transformation/Handling of temporaries/local arrays

    - `parametrise`: parametrising the array dimensions to make the vertical dimension
      a compile-time constant
    - `hoist`: host side hoisting of (relevant) arrays
mode: str
    Mode/language to target

    - `CUF` - CUDA Fortran
    - `CUDA` - CUDA C
    - `HIP` - HIP
"""
SCCLowLevelCuf = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCRevectorTransformation,
        LowerBlockIndexTransformation,
        InjectBlockIndexTransformation,
        LowerBlockLoopTransformation,
        SccLowLevelLaunchConfiguration,
        SccLowLevelDataOffload,
    )
)

"""
The Single Column Coalesced low-level GPU via CUDA-Fortran (SCC-CUF)
handling temporaries via parametrisation.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCLowLevelCuf`.

In addition, this pipeline will invoke
:any:`ParametriseTransformation` to parametrise relevant array
dimensions to allow having temporary arrays.

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the horizontal data dimension and iteration space.
block_dim : :any:`Dimension`
    Optional ``Dimension`` object to define the blocking dimension
    to use for hoisted column arrays if hoisting is enabled.
directive : string or None
    Directives flavour to use for parallelism annotations; either
    ``'openacc'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
derived_types: tuple
    List of relevant derived types
transformation_type : str
    Kind of transformation/Handling of temporaries/local arrays

    - `parametrise`: parametrising the array dimensions to make the vertical dimension
      a compile-time constant
    - `hoist`: host side hoisting of (relevant) arrays
mode: str
    Mode/language to target

    - `CUF` - CUDA Fortran
    - `CUDA` - CUDA C
    - `HIP` - HIP
dic2p: dict
    Dictionary of variable names and corresponding values to be parametrised.
"""
SCCLowLevelCufParametrise = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCRevectorTransformation,
        LowerBlockIndexTransformation,
        InjectBlockIndexTransformation,
        LowerBlockLoopTransformation,
        SccLowLevelLaunchConfiguration,
        SccLowLevelDataOffload,
        ParametriseTransformation
    )
)

"""
The Single Column Coalesced low-level GPU via CUDA-Fortran (SCC-CUF)
handling temporaries via hoisting.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCLowLevelCuf`.

In addition, this pipeline will invoke
:any:`HoistTemporaryArraysAnalysis` and
:any:`HoistTemporaryArraysDeviceAllocatableTransformation`
to hoist temporary arrays.

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the horizontal data dimension and iteration space.
block_dim : :any:`Dimension`
    Optional ``Dimension`` object to define the blocking dimension
    to use for hoisted column arrays if hoisting is enabled.
directive : string or None
    Directives flavour to use for parallelism annotations; either
    ``'openacc'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
derived_types: tuple
    List of relevant derived types
transformation_type : str
    Kind of transformation/Handling of temporaries/local arrays

    - `parametrise`: parametrising the array dimensions to make the vertical dimension
      a compile-time constant
    - `hoist`: host side hoisting of (relevant) arrays
mode: str
    Mode/language to target

    - `CUF` - CUDA Fortran
    - `CUDA` - CUDA C
    - `HIP` - HIP
"""
SCCLowLevelCufHoist = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCRevectorTransformation,
        LowerBlockIndexTransformation,
        InjectBlockIndexTransformation,
        LowerBlockLoopTransformation,
        SccLowLevelLaunchConfiguration,
        SccLowLevelDataOffload,
        HoistTemporaryArraysAnalysis,
        HoistTemporaryArraysDeviceAllocatableTransformation
    )
)

"""
The Single Column Coalesced low-level GPU via low-level C-style
kernel language (CUDA, HIP, ...) handling temporaries via parametrisation.

This tranformation will convert kernels with innermost vectorisation
along a common horizontal dimension to a GPU-friendly loop-layout via
loop inversion and local array variable demotion. The resulting kernel
remains "vector-parallel", but with the ``horizontal`` loop as the
outermost iteration dimension (as far as data dependencies
allow). This allows local temporary arrays to be demoted to scalars,
where possible.

Kernels are specified via e.g., ``'__global__'`` and the number of threads that
execute the kernel for a given call is specified via the chevron syntax.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:
1. :any:`InlineTransformation` - Inline constants and elemental
   functions.
2. :any:`GlobalVariableAnalysis` - Analysis of global variables
3. :any:`GlobalVarHoistTransformation` - Hoist global variables
   to the driver.
4. :any:`DerivedTypeArgumentsTransformation` - Flatten derived types/
   remove derived types from procedure signatures by replacing the
   (relevant) derived type arguments by its member variables.
5. :any:`SCCBaseTransformation` - Ensure utility variables and resolve
   problematic code constructs.
6. :any:`SCCDevectorTransformation` - Remove horizontal vector loops.
7. :any:`SCCDemoteTransformation` - Demote local temporary array
   variables where appropriate.
8. :any:`SCCRevectorTransformation` - Re-insert the vecotr loops outermost,
   according to identified vector sections.
9. :any:`LowerBlockIndexTransformation` - Lower the block index (for
   array argument definitions).
10. :any:`InjectBlockIndexTransformation` - Complete the previous step
   and inject the block index for the relevant arrays.
11. :any:`LowerBlockLoopTransformation` - Lower the block loop
   from driver to kernel(s).
12. :any:`SCCLowLevelLaunchConfiguration` - Create launch configuration
   and related things.
13. :any:`SCCLowLevelDataOffload` - Create/handle data offload
   and related things.
14. :any:`ParametriseTransformation` - Parametrise according to ``dic2p``.

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the horizontal data dimension and iteration space.
block_dim : :any:`Dimension`
    Optional ``Dimension`` object to define the blocking dimension
    to use for hoisted column arrays if hoisting is enabled.
directive : string or None
    Directives flavour to use for parallelism annotations; either
    ``'openacc'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
derived_types: tuple
    List of relevant derived types
transformation_type : str
    Kind of transformation/Handling of temporaries/local arrays

    - `parametrise`: parametrising the array dimensions to make the vertical dimension
      a compile-time constant
    - `hoist`: host side hoisting of (relevant) arrays
mode: str
    Mode/language to target

    - `CUF` - CUDA Fortran
    - `CUDA` - CUDA C
    - `HIP` - HIP
dic2p: dict
    Dictionary of variable names and corresponding values to be parametrised.
"""
SCCLowLevelParametrise = partial(
    Pipeline, classes=(
        InlineTransformation,
        GlobalVariableAnalysis,
        GlobalVarHoistTransformation,
        DerivedTypeArgumentsTransformation,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCRevectorTransformation,
        LowerBlockIndexTransformation,
        InjectBlockIndexTransformation,
        LowerBlockLoopTransformation,
        SccLowLevelLaunchConfiguration,
        SccLowLevelDataOffload,
        ParametriseTransformation
    )
)

"""
The Single Column Coalesced low-level GPU via low-level C-style
kernel language (CUDA, HIP, ...) handling temporaries via parametrisation.

This tranformation will convert kernels with innermost vectorisation
along a common horizontal dimension to a GPU-friendly loop-layout via
loop inversion and local array variable demotion. The resulting kernel
remains "vector-parallel", but with the ``horizontal`` loop as the
outermost iteration dimension (as far as data dependencies
allow). This allows local temporary arrays to be demoted to scalars,
where possible.

Kernels are specified via e.g., ``'__global__'`` and the number of threads that
execute the kernel for a given call is specified via the chevron syntax.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:
1. :any:`InlineTransformation` - Inline constants and elemental
   functions.
2. :any:`GlobalVariableAnalysis` - Analysis of global variables
3. :any:`GlobalVarHoistTransformation` - Hoist global variables
   to the driver.
4. :any:`DerivedTypeArgumentsTransformation` - Flatten derived types/
   remove derived types from procedure signatures by replacing the
   (relevant) derived type arguments by its member variables.
5. :any:`SCCBaseTransformation` - Ensure utility variables and resolve
   problematic code constructs.
6. :any:`SCCDevectorTransformation` - Remove horizontal vector loops.
7. :any:`SCCDemoteTransformation` - Demote local temporary array
   variables where appropriate.
8. :any:`SCCRevectorTransformation` - Re-insert the vecotr loops outermost,
   according to identified vector sections.
9. :any:`LowerBlockIndexTransformation` - Lower the block index (for
   array argument definitions).
10. :any:`InjectBlockIndexTransformation` - Complete the previous step
   and inject the block index for the relevant arrays.
11. :any:`LowerBlockLoopTransformation` - Lower the block loop
   from driver to kernel(s).
12. :any:`SCCLowLevelLaunchConfiguration` - Create launch configuration
   and related things.
13. :any:`SCCLowLevelDataOffload` - Create/handle data offload
   and related things.
14. :any:`HoistTemporaryArraysAnalysis` - Analysis part of hoisting.
15. :any:`HoistTemporaryArraysPragmaOffloadTransformation` - Syntesis
    part of hoisting.

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the horizontal data dimension and iteration space.
block_dim : :any:`Dimension`
    Optional ``Dimension`` object to define the blocking dimension
    to use for hoisted column arrays if hoisting is enabled.
directive : string or None
    Directives flavour to use for parallelism annotations; either
    ``'openacc'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
derived_types: tuple
    List of relevant derived types
transformation_type : str
    Kind of transformation/Handling of temporaries/local arrays

    - `parametrise`: parametrising the array dimensions to make the vertical dimension
      a compile-time constant
    - `hoist`: host side hoisting of (relevant) arrays
mode: str
    Mode/language to target

    - `CUF` - CUDA Fortran
    - `CUDA` - CUDA C
    - `HIP` - HIP
"""
SCCLowLevelHoist = partial(
    Pipeline, classes=(
        InlineTransformation,
        GlobalVariableAnalysis,
        GlobalVarOffloadTransformation,
        GlobalVarHoistTransformation,
        DerivedTypeArgumentsTransformation,
        ArgumentArrayShapeAnalysis,
        ExplicitArgumentArrayShapeTransformation,
        LowerConstantArrayIndex,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCRevectorTransformation,
        LowerBlockIndexTransformation,
        InjectBlockIndexTransformation,
        LowerBlockLoopTransformation,
        SccLowLevelLaunchConfiguration,
        SccLowLevelDataOffload,
        # ParametriseArrayDimsTransformation,
        HoistTemporaryArraysAnalysis,
        HoistTemporaryArraysPragmaOffloadTransformation
    )
)
