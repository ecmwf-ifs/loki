# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

from loki.batch import Pipeline

from loki.transformations.hoist_variables import HoistTemporaryArraysAnalysis
from loki.transformations.pool_allocator import TemporariesPoolAllocatorTransformation
from loki.transformations.raw_stack_allocator import TemporariesRawStackTransformation
from loki.transformations.raw_stack_allocator_new import TemporariesRawStackNewTransformation

from loki.transformations.single_column.base import SCCBaseTransformation
from loki.transformations.single_column.annotate import SCCAnnotateTransformation
from loki.transformations.single_column.hoist import SCCHoistTemporaryArraysTransformation
from loki.transformations.single_column.vector import (
    SCCDevectorTransformation, SCCDemoteTransformation,
    SCCVecRevectorTransformation, SCCSeqRevectorTransformation
)
from loki.transformations.single_column.vertical import SCCFuseVerticalLoops
from loki.transformations.pragma_model import PragmaModelTransformation

__all__ = [
    'SCCVectorPipeline', 'SCCVVectorPipeline', 'SCCSVectorPipeline',
    'SCCHoistPipeline', 'SCCVHoistPipeline', 'SCCSHoistPipeline',
    'SCCStackPipeline', 'SCCVStackPipeline', 'SCCSStackPipeline',
    'SCCRawStackPipeline', 'SCCVRawStackPipeline', 'SCCSRawStackPipeline',
    'SCCRawStackNewPipeline', 'SCCVRawStackNewPipeline', 'SCCSRawStackNewPipeline'
]

SCCVVectorPipeline = partial(
    Pipeline, classes=(
        SCCFuseVerticalLoops,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCVecRevectorTransformation,
        SCCAnnotateTransformation,
        PragmaModelTransformation
    )
)
"""
The basic Single Column Coalesced (SCC) transformation with
vector-level kernel parallelism.

This tranformation will convert kernels with innermost vectorisation
along a common horizontal dimension to a GPU-friendly loop-layout via
loop inversion and local array variable demotion. The resulting kernel
remains "vector-parallel", but with the ``horizontal`` loop as the
outermost iteration dimension (as far as data dependencies
allow). This allows local temporary arrays to be demoted to scalars,
where possible.

The outer "driver" loop over blocks is used as the secondary dimension
of parallelism, where the outher data indexing dimension
(``block_dim``) is resolved in the first call to a "kernel"
routine. This is equivalent to a so-called "gang-vector" parallelisation
scheme.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:
1. :any:`SCCBaseTransformation` - Ensure utility variables and resolve
   problematic code constructs.
2. :any:`SCCDevectorTransformation` - Remove horizontal vector loops.
3. :any:`SCCDemoteTransformation` - Demote local temporary array
   variables where appropriate.
4. :any:`SCCVecRevectorTransformation` - Re-insert the vector loops outermost,
   according to identified vector sections.
5. :any:`SCCAnnotateTransformation` - Annotate loops according to
   programming model (``directive``).

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
"""

# alias for backwards compability
SCCVectorPipeline = SCCVVectorPipeline

SCCSVectorPipeline = partial(
    Pipeline, classes=(
        SCCFuseVerticalLoops,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCSeqRevectorTransformation,
        SCCAnnotateTransformation,
        PragmaModelTransformation
    )
)
"""
The basic Single Column Coalesced (SCC) transformation with
sequential kernels.

This tranformation will convert kernels with innermost vectorisation
along a common horizontal dimension to a GPU-friendly loop-layout via
loop inversion and local array variable demotion. The resulting kernel
becomes sequential as the ``horizontal`` loop is hoisted to the driver
and the loop index becomes an argument to the kernel(s).
Moreover, this allows local temporary arrays to be demoted to scalars,
where possible.

The outer "driver" loop over blocks is used as the secondary dimension
of parallelism, where the outher data indexing dimension
(``block_dim``) is resolved in the first call to a "kernel"
routine. This is equivalent to a so-called "gang-vector" parallelisation
scheme.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:
1. :any:`SCCBaseTransformation` - Ensure utility variables and resolve
   problematic code constructs.
2. :any:`SCCDevectorTransformation` - Remove horizontal vector loops.
3. :any:`SCCDemoteTransformation` - Demote local temporary array
   variables where appropriate.
4. :any:`SCCSeqRevectorTransformation` - Re-insert the vector loops outermost,
   according to identified vector sections.
5. :any:`SCCAnnotateTransformation` - Annotate loops according to
   programming model (``directive``).

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
"""

SCCVHoistPipeline = partial(
    Pipeline, classes=(
        SCCFuseVerticalLoops,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCVecRevectorTransformation,
        HoistTemporaryArraysAnalysis,
        SCCHoistTemporaryArraysTransformation,
        SCCAnnotateTransformation,
        PragmaModelTransformation
    )
)
"""
SCC-style transformation with "vector-parallel" kernels
that additionally hoists local temporary
arrays that cannot be demoted to the outer driver call.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCVVectorPipeline`

In addition, this pipeline will invoke
:any:`HoistTemporaryArraysAnalysis` and
:any:`SCCHoistTemporaryArraysTransformation` before the final
annotation step to hoist multi-dimensional local temporary array
variables to the "driver" routine, where they will be allocated on
device and passed down as arguments.

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
dim_vars: tuple of str, optional
    Variables to be within the dimensions of the arrays to be
    hoisted. If not provided, no checks will be done for the array
    dimensions in :any:`HoistTemporaryArraysAnalysis`.
"""

SCCSHoistPipeline = partial(
    Pipeline, classes=(
        SCCFuseVerticalLoops,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCSeqRevectorTransformation,
        HoistTemporaryArraysAnalysis,
        SCCHoistTemporaryArraysTransformation,
        SCCAnnotateTransformation,
        PragmaModelTransformation
    )
)
"""
SCC-style transformation with sequential kernels
that additionally hoists local temporary
arrays that cannot be demoted to the outer driver call.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCSVectorPipeline`

In addition, this pipeline will invoke
:any:`HoistTemporaryArraysAnalysis` and
:any:`SCCHoistTemporaryArraysTransformation` before the final
annotation step to hoist multi-dimensional local temporary array
variables to the "driver" routine, where they will be allocated on
device and passed down as arguments.

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
dim_vars: tuple of str, optional
    Variables to be within the dimensions of the arrays to be
    hoisted. If not provided, no checks will be done for the array
    dimensions in :any:`HoistTemporaryArraysAnalysis`.
"""

# alias for backwards compability
SCCHoistPipeline = SCCVHoistPipeline

SCCVStackPipeline = partial(
    Pipeline, classes=(
        SCCFuseVerticalLoops,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCVecRevectorTransformation,
        SCCAnnotateTransformation,
        TemporariesPoolAllocatorTransformation,
        PragmaModelTransformation
    )
)
"""
SCC-style transformation with "vector-parallel" kernels
that additionally pre-allocates a "stack"
pool allocator and associates local arrays with preallocated memory.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCVVectorPipeline`

In addition, this pipeline will invoke
:any:`TemporariesPoolAllocatorTransformation` to back the remaining
locally allocated arrays from a "stack" pool allocator that is
pre-allocated in the driver routine and passed down via arguments.

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
check_bounds : bool, optional
    Insert bounds-checks in the kernel to make sure the allocated
    stack size is not exceeded (default: `True`)
"""

# alias for backwards compability
SCCStackPipeline = SCCVStackPipeline

SCCSStackPipeline = partial(
    Pipeline, classes=(
        SCCFuseVerticalLoops,
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCSeqRevectorTransformation,
        SCCAnnotateTransformation,
        TemporariesPoolAllocatorTransformation,
        PragmaModelTransformation
    )
)
"""
SCC-style transformation with sequential kernels
that additionally pre-allocates a "stack"
pool allocator and associates local arrays with preallocated memory.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCSVectorPipeline`

In addition, this pipeline will invoke
:any:`TemporariesPoolAllocatorTransformation` to back the remaining
locally allocated arrays from a "stack" pool allocator that is
pre-allocated in the driver routine and passed down via arguments.

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
check_bounds : bool, optional
    Insert bounds-checks in the kernel to make sure the allocated
    stack size is not exceeded (default: `True`)
"""

SCCVRawStackPipeline = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCVecRevectorTransformation,
        SCCAnnotateTransformation,
        HoistTemporaryArraysAnalysis,
        SCCHoistTemporaryArraysTransformation,
        TemporariesRawStackTransformation,
        PragmaModelTransformation
    )
)
"""
SCC-style transformation with "vector-parallel" kernels
that additionally pre-allocates a "stack"
pool allocator and replaces local temporaries with indexed sub-arrays
of this preallocated array.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCVectorPipeline`

In addition, this pipeline will invoke
:any:`TemporariesRawStackTransformation` to back the remaining
locally allocated arrays from a "stack" pool allocator that is
pre-allocated in the driver routine and passed down via arguments.

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
check_bounds : bool, optional
    Insert bounds-checks in the kernel to make sure the allocated
    stack size is not exceeded (default: `True`)
driver_horizontal : str, optional
    Override string if a separate variable name should be used for the
    horizontal when allocating the stack in the driver.
"""

# alias for backwards compability
SCCRawStackPipeline = SCCVRawStackPipeline

SCCSRawStackPipeline = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCSeqRevectorTransformation,
        SCCAnnotateTransformation,
        HoistTemporaryArraysAnalysis,
        SCCHoistTemporaryArraysTransformation,
        TemporariesRawStackTransformation,
        PragmaModelTransformation
    )
)
"""
SCC-style transformation with sequential kernels
that additionally pre-allocates a "stack"
pool allocator and replaces local temporaries with indexed sub-arrays
of this preallocated array.

For details of the kernel and driver-side transformations, please
refer to :any:`SCCVectorPipeline`

In addition, this pipeline will invoke
:any:`TemporariesRawStackTransformation` to back the remaining
locally allocated arrays from a "stack" pool allocator that is
pre-allocated in the driver routine and passed down via arguments.

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
    ``'openacc'``, ``'omp-gpu'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
check_bounds : bool, optional
    Insert bounds-checks in the kernel to make sure the allocated
    stack size is not exceeded (default: `True`)
driver_horizontal : str, optional
    Override string if a separate variable name should be used for the
    horizontal when allocating the stack in the driver.
"""

SCCVRawStackNewPipeline = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCVecRevectorTransformation,
        SCCAnnotateTransformation,
        HoistTemporaryArraysAnalysis,
        SCCHoistTemporaryArraysTransformation,
        TemporariesRawStackNewTransformation,
        PragmaModelTransformation
    )
)

SCCRawStackNewPipeline = SCCVRawStackNewPipeline

SCCSRawStackNewPipeline = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCSeqRevectorTransformation,
        SCCAnnotateTransformation,
        HoistTemporaryArraysAnalysis,
        SCCHoistTemporaryArraysTransformation,
        TemporariesRawStackNewTransformation,
        PragmaModelTransformation
    )
)
