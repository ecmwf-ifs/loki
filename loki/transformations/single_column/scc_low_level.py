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
        LowerBlockIndexTransformation, LowerBlockLoopTransformation
)
from loki.transformations.transform_derived_types import DerivedTypeArgumentsTransformation
from loki.transformations.data_offload import (
    GlobalVariableAnalysis, GlobalVarHoistTransformation
)
from loki.transformations.parametrise import ParametriseTransformation
from loki.transformations.inline import (
    inline_constant_parameters, inline_elemental_functions
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
        # inline_elemental_kernel(routine, **kwargs)
        role = kwargs['role']

        if role == 'kernel':

            inline_constant_parameters(routine, external_only=True)
            inline_elemental_functions(routine)

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

SCCLowLevelHoist = partial(
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
        HoistTemporaryArraysAnalysis,
        HoistTemporaryArraysPragmaOffloadTransformation
    )
)
