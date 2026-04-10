# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

from loki.batch import Pipeline

from loki.transformations.cpu.tmp import CPUBaseTransformation
from loki.transformations.cpu.promote import CPUPromoteTransformation

from loki.transformations.cpu.inline_calls import InlineCallSiteForVectorisation
from loki.transformations.cpu.hoist_io import HoistWriteFromLoop
from loki.transformations.cpu.safe_denominator import SafeDenominatorGuard
from loki.transformations.cpu.merge_conditionals import ConditionalFPGuardToMerge
from loki.transformations.cpu.merge_to_explicit_mask import MergeToExplicitMask
from loki.transformations.cpu.loop_split import SplitLoopForVectorisation
from loki.transformations.cpu.simd_pragmas import InsertSIMDPragmaDirectives
from loki.transformations.cpu.outline_sections import ExtractOutlinePhysicsSection


__all__ = [
    'CPUBasePipeline',
    'CPUVectorisationPipeline',
    'CPUSafeDenomPipeline',
    'CPUMergePipeline',
    'CPUSafeDenomMergePipeline',
    'CPUMaskingPipeline',
    'CPULoopSplitPipeline',
    'CPUSIMDPipeline',
]


CPUBasePipeline = partial(
    Pipeline, classes=(
        CPUBaseTransformation,
        CPUPromoteTransformation
    )
)

CPUVectorisationPipeline = partial(
    Pipeline, classes=(
        InlineCallSiteForVectorisation,
        HoistWriteFromLoop,
        SafeDenominatorGuard,
        ConditionalFPGuardToMerge,
        MergeToExplicitMask,
        # SplitLoopForVectorisation,
        InsertSIMDPragmaDirectives,
        # ExtractOutlinePhysicsSection,
    )
)
"""
CPU vectorisation pipeline targeting Intel compiler optrpt remarks.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:

1. :any:`InlineCallSiteForVectorisation` (T5) — Inline small
   subroutine calls inside horizontal loops to remove call-site
   vectorisation blockers (#15543).
2. :any:`HoistWriteFromLoop` (T3) — Replace WRITE/PRINT inside
   horizontal loops with boolean flags and post-loop diagnostics
   to remove I/O vectorisation blockers (#15344).
3. :any:`SafeDenominatorGuard` (T2) — Wrap dangerous FP operands
   (division denominator, SQRT/LOG/EXP args) with MAX/MIN/TINY
   clamps so unconditional evaluation is safe.
4. :any:`ConditionalFPGuardToMerge` (T1) — Convert IF/ELSE blocks
   with pure assignments to MERGE intrinsic calls, removing
   data-dependent branches (#15326).
5. :any:`MergeToExplicitMask` (T1b) — Rewrite ``MERGE`` calls into
   explicit arithmetic masking using integer ``MERGE(1, 0, mask)``
   so that compilers can vectorise under ``-fp-model strict``.
6. :any:`SplitLoopForVectorisation` (T7) — Split horizontal loops
   that still contain a mix of vectorisable and non-vectorisable
   statements into separate loops.  Scalar variables crossing
   split boundaries are promoted to arrays.
7. :any:`InsertSIMDPragmaDirectives` (T6) — Insert ``!DIR$ SIMD``
   / ``!$OMP SIMD`` pragmas on horizontal loops (#15541).
8. :any:`ExtractOutlinePhysicsSection` (T4) — Split large routines
   into smaller subroutines to reduce register pressure and
   compilation-time budget issues (#15532).

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used
    in code to define the horizontal data dimension and iteration space.
"""


# =====================================================================
# Individual CPU transformation pipelines for independent testing
# =====================================================================
# Each pipeline isolates a single transformation (or a minimal
# dependency chain) so that its impact on vectorisation, performance,
# and numerical validity can be assessed independently.

CPUSafeDenomPipeline = partial(
    Pipeline, classes=(
        SafeDenominatorGuard,
    )
)
"""
Guard dangerous FP operands (division denominator, SQRT/LOG/EXP args)
with MAX/MIN/TINY clamps so unconditional evaluation is safe.

Parameters
----------
horizontal : :any:`Dimension`
"""

CPUMergePipeline = partial(
    Pipeline, classes=(
        ConditionalFPGuardToMerge,
    )
)
"""
Convert IF/ELSE blocks with pure assignments to MERGE intrinsic calls,
removing data-dependent branches.  Without :any:`SafeDenominatorGuard`
running first, the MERGE will unconditionally evaluate both branches —
including potentially unsafe FP operations.

Parameters
----------
horizontal : :any:`Dimension`
"""

CPUSafeDenomMergePipeline = partial(
    Pipeline, classes=(
        SafeDenominatorGuard,
        ConditionalFPGuardToMerge,
    )
)
"""
Natural pair: first guard FP operands, then convert IF/ELSE to MERGE.
This is the safe combination — denominators are clamped before branches
are removed.

Parameters
----------
horizontal : :any:`Dimension`
"""

CPUMaskingPipeline = partial(
    Pipeline, classes=(
        SafeDenominatorGuard,
        ConditionalFPGuardToMerge,
        MergeToExplicitMask,
    )
)
"""
Full T1→T2→T3 chain: guard FP operands, convert IF/ELSE to MERGE,
then rewrite MERGE into explicit arithmetic masking.  Requires
``fp_strict=True`` on :any:`MergeToExplicitMask` to be active.

Parameters
----------
horizontal : :any:`Dimension`
fp_strict : bool
    Must be ``True`` for :any:`MergeToExplicitMask` to activate.
"""

CPULoopSplitPipeline = partial(
    Pipeline, classes=(
        SplitLoopForVectorisation,
    )
)
"""
Split horizontal loops that contain a mix of vectorisable and
non-vectorisable statements into separate loops.

Parameters
----------
horizontal : :any:`Dimension`
"""

CPUSIMDPipeline = partial(
    Pipeline, classes=(
        InsertSIMDPragmaDirectives,
    )
)
"""
Insert ``!$OMP SIMD`` pragmas on horizontal loops.

Parameters
----------
horizontal : :any:`Dimension`
"""
