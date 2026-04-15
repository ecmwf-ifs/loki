# Remaining Bugs Investigation — SCCSmallKernelsPipeline

This document collects the root-cause analysis, proposed fix approaches, test
sketches, and classification for all remaining (unfixed) bug categories found
when comparing the Loki-generated output against the manually corrected
`working_ifs_sp/` reference.

**Categories already fixed**: 1, 2, 4, 7, 8 (plus cosmetic diffs which require
no code changes).

**Categories covered in this document**: 3, 5, 6, 9, 10, 11, 12.

---

## Category 3 — Stack size (ISTSZ) computation is wrong or missing

### Affected files
- `ecphys_setup_layer` — Missing `ISTSZ` entirely
- `lacdyn` — Missing `ISTSZ` entirely
- `lapineb_parallel` — No `ISTSZ`/`ZSTACK` at all; `YLSTACK_L` declared but
  never initialised → garbage stack pointer

### Root cause

The pool allocator (`TemporariesPoolAllocatorPerDrvLoopTransformation`) computes
`ISTSZ` via `_determine_stack_size()`, which looks for `CallStatement` nodes
inside the driver loop body that match a successor in `successor_map`. The
stack size is the `MAX()` of all successor stack sizes.

**Problem 1 (ecphys_setup_layer, lacdyn — kernel role):**
When `role == 'kernel'`, `transform_subroutine` still calls `find_driver_loops`
on the routine body (line 240 of `pool_allocator_per_drv_loop.py`). If it
finds a loop that looks like a driver loop (because it contains calls to
target routines), it enters the per-driver-loop branch and calls
`_determine_stack_size` scoped to that loop's body. However, in the kernel
context the successors' `trafo_data` may not contain a matching `stack_size`
entry (because the successor was processed as a kernel too, and its stack
size was stored at the routine level, not per loop). The result is that
`_determine_stack_size` finds no matching successors in the loop body →
returns `Literal(0)` → `ISTSZ = 0`.

**Problem 2 (lapineb_parallel — driver role):**
`find_driver_loops` returns an empty list for `lapineb_parallel` because the
block loop was already eliminated by `LowerBlockLoopTransformation` (which
runs in a previous pipeline step). The block loop was lowered into the
kernel, so there's no loop left for `find_driver_loops` to find. Yet the
pool allocator was already configured to generate `YLSTACK_L`/`YLSTACK_U`
declarations (added by prior kernel-level processing). With no driver loop
found, the driver-level ISTSZ/ZSTACK allocation code never runs, leaving
`YLSTACK_L` uninitialised.

### Source code reference
- `pool_allocator_per_drv_loop.py:240-268` — `transform_subroutine`
- `pool_allocator_per_drv_loop.py:531-610` — `_determine_stack_size`

### Proposed fix

**For Problem 1**: In `transform_subroutine`, skip the per-driver-loop
ISTSZ/ZSTACK setup when `role == 'kernel'`. Kernels should not get their own
pool allocator setup for loops that happen to contain calls to target
routines — that's the driver's responsibility. The kernel-level pool
allocator (`apply_pool_allocator_to_temporaries`) already handles replacing
kernel temporaries with Cray pointers.

Concretely: guard the `if driver_loops:` block (lines 242-258) with
`if role == 'driver':` or add a separate kernel path that only does
`_determine_stack_size` for aggregation purposes (without creating ISTSZ/
ZSTACK variables).

**For Problem 2**: The interaction between `LowerBlockLoopTransformation` and
the pool allocator needs revisiting. When the block loop is lowered, the pool
allocator must still be able to find the "driver loop" (now inside the
kernel) or the driver must generate ISTSZ based on successor trafo_data
without needing a local driver loop. A pragmatic fix: if `find_driver_loops`
returns empty for a driver, fall back to computing ISTSZ from the aggregated
successor trafo_data at the routine level (not per-loop).

### Classification
- **Loki code change** (pool_allocator_per_drv_loop.py)

### Test sketch

```python
def test_cat3_kernel_no_spurious_istsz():
    """
    When a kernel contains a loop with calls to target routines,
    the pool allocator must NOT generate ISTSZ/ZSTACK at the kernel
    level — that's the driver's job.
    """
    # 1. Create a driver → kernel → sub_kernel hierarchy
    # 2. The kernel has a DO JFLD=1,N loop containing CALL sub_kernel(...)
    # 3. Apply the full pipeline
    # 4. Assert kernel does NOT contain 'ISTSZ' variable
    # 5. Assert driver DOES contain 'ISTSZ' with correct size
```

```python
def test_cat3_driver_without_block_loop():
    """
    When a driver's block loop has been lowered (eliminated by
    LowerBlockLoopTransformation), the pool allocator must still
    generate ISTSZ/ZSTACK based on successor trafo_data.
    """
    # 1. Create driver with block loop that gets lowered
    # 2. Apply pipeline including LowerBlockLoop + PoolAllocator
    # 3. Assert ISTSZ and ZSTACK are declared and initialised
    # 4. Assert YLSTACK_L is assigned (not uninitialised)
```

---

## Category 5 — Rank mismatch / array reshaping for VERDISINT calls

### Affected files
- `cpg_gp_hyd` — `YDVARS%EDOT%T0_FIELD` is 3D (NPROMA x NFLEVG x NGPBLKS)
  but after block index injection becomes `T0_FIELD(:,:,:,local_IBL)` — 4
  subscripts on a 3D array

### Root cause

`InjectBlockIndexTransformation` adds a block index dimension to array
arguments when it detects a rank mismatch between the actual argument in
the caller and the dummy argument in the callee. The logic at
`block_index_transformations.py:749` checks:

```python
if isinstance(arg, Array) and self.get_call_arg_rank(call_arg, ...) > len(arg.shape):
```

For `EDOT`, the variable is stored as `VARIABLE_3RB` type with a degenerate
`NFLEVG` dimension (it's really a 2D field: NPROMA x NGPBLKS, but declared
3D for Fortran sequence-association reasons). When the block index is
appended, the result is 4 subscripts on a 3D array.

The deeper issue is that the original IFS code relies on Fortran sequence
association (passing a 3D array slice to a 2D dummy argument), and Loki's
block-index injection doesn't account for this pattern. The transformation
assumes `actual_rank - 1 == dummy_rank` means "append block index", but
doesn't handle cases where the rank difference is due to degenerate
intermediate dimensions.

### Source code reference
- `block_index_transformations.py:744-755` — dimension update in `process_kernel`
- `block_index_transformations.py:927-938` — dimension update in `process_driver`

### Proposed fix

**Option A (Loki code change):** Before appending the block index, check if
the rank mismatch is exactly 1. If the actual argument already has more
dimensions than the dummy + 1, this is a sequence-association pattern and
the block index should NOT be naively appended. Instead, generate a
reshape/temporary or skip the injection for that argument.

**Option B (IFS source change):** Modify the IFS source so that `EDOT` fields
are declared with the correct rank, avoiding reliance on sequence association.
This would require changes to the `YDVARS` derived-type definition and all
call sites — a larger change but architecturally cleaner.

**Option C (Hybrid):** For the specific case where `actual_rank > dummy_rank + 1`,
emit a warning and skip block-index injection for that argument. This is a
"safe default" — the generated code will be incomplete but won't be actively
wrong (compiler will catch it vs. silently wrong results).

### Classification
- **IFS source change** preferred (Option B) for correctness
- **Loki code change** (Option A or C) as fallback

### Test sketch

```python
def test_cat5_rank_mismatch_no_extra_block_index():
    """
    When a 3D array is passed to a 2D dummy argument (sequence
    association), InjectBlockIndexTransformation must not append
    a 4th subscript.
    """
    # 1. Create kernel with 2D dummy argument: REAL :: FIELD(KLON, KLEV)
    # 2. Driver passes 3D actual: FIELD3D(:,:,IBL) where FIELD3D is (KLON, KLEV, NGPBLKS)
    #    but FIELD3D is actually declared as (KLON, 1, NGPBLKS) — degenerate middle dim
    # 3. Apply InjectBlockIndexTransformation
    # 4. Assert the call argument is NOT FIELD3D(:,:,:,IBL) (4 subscripts)
    # 5. Assert instead it remains FIELD3D(:,:,IBL) or uses correct reshaping
```

---

## Category 6 — Missing `IBL` argument passed to kernel calls in host path

### Affected files
- `cpg_0_parallel` — Missing `IBL=IBL` in non-GPU host-path call
- `cpg_2_parallel` — Same
- `lapineb_parallel` — Same

### Root cause

`LowerBlockIndexTransformation.process_driver` (line 806+) only modifies
calls found inside `find_driver_loops()` results. `find_driver_loops` looks
for loops containing calls to target routines marked with
`!$loki small-kernels`. In a driver like `cpg_0_parallel`, the code has
two paths:

1. **GPU path**: block loop `DO IBL=1,NGPBLKS` containing
   `!$loki small-kernels` + `CALL KERNEL_LOKI(...)` — this IS found by
   `find_driver_loops`
2. **Host/OMP path**: `!$OMP PARALLEL DO` + `DO IBL=1,NGPBLKS` +
   `CALL KERNEL(...)` (the original, non-`_LOKI` routine) — this is NOT
   found because the call target doesn't match `targets` (which contains
   the `_LOKI` variant name)

After `LowerBlockLoopTransformation` eliminates the GPU-path block loop,
the kernel now needs `IBL` as an argument. But the host-path call to the
original (non-`_LOKI`) routine was never updated to pass `IBL`.

### Source code reference
- `block_index_transformations.py:806-973` — `process_driver`
- `block_index_transformations.py:824-833` — only processes calls inside
  `find_driver_loops` results

### Proposed fix

**Option A (Loki code change):** After processing the GPU-path calls, scan
the entire routine body for ANY call to the same base routine name (with or
without `_LOKI` suffix) and add `IBL=IBL` if the callee now has `IBL` in
its argument list. This is a second pass that catches the host-path calls.

**Option B (Loki code change):** In `process_driver`, also look for calls
in OMP parallel regions to the same routine. This could be done by extending
`find_driver_loops` to also return OMP-annotated block loops, or by doing a
separate search for `CallStatement` nodes outside the GPU-path loops.

**Option C (IFS source change):** Add `!$loki small-kernels` pragmas to the
host-path calls too. However, this was tried and didn't work because
`find_driver_loops` still doesn't pick up the OMP loop as a "driver loop"
for `LowerBlockIndex` purposes.

### Classification
- **Loki code change** (block_index_transformations.py)

### Test sketch

```python
def test_cat6_host_path_gets_ibl_argument():
    """
    When a driver has both a GPU-path block loop (with !$loki small-kernels)
    and a host-path OMP block loop calling the same routine,
    LowerBlockIndexTransformation must add IBL=IBL to BOTH calls.
    """
    # 1. Create driver with two block loops:
    #    - GPU path: DO IBL=1,NGPBLKS ... !$loki small-kernels ... CALL kernel(...)
    #    - Host path: !$OMP PARALLEL DO ... DO IBL=1,NGPBLKS ... CALL kernel(...)
    # 2. Apply LowerBlockIndexTransformation
    # 3. Assert the GPU-path call has IBL in kwargs
    # 4. Assert the host-path call ALSO has IBL in kwargs
```

---

## Category 9 — Missing keyword arguments (YDCPG_BNDS, YDCPG_OPTS) in calls

### Affected files
- `lassie` — `SITNU_GP_LOKI`, `SIGAM_GP_LOKI` calls missing
  `YDCPG_BNDS=local_YDCPG_BNDS, YDCPG_OPTS=YDCPG_OPTS` kwargs
- `cpg_gp_hyd` — `GPCTY_EXPL_LOKI`, `GPGEO_EXPL_LOKI`, `GPGRGEO_EXPL_LOKI`
  missing same kwargs

### Root cause

`LowerBlockIndexTransformation.process_kernel` (line 640+) collects
`relevant_vars` from the kernel's `LowerBlockIndex` trafo_data, which was
propagated from the driver. The `relevant_vars` set is built from:
1. Driver loop bounds (lower, upper, step)
2. Variables used in driver loop header assignments

In `lassie`, the driver loop header contains assignments like
`bnds%kidia = 1`, `bnds%kfdia = min(...)` etc. These derived-type
components are collected as `relevant_vars`. But when `process_kernel` tries
to pass them to sub-kernel calls, the logic at line 676-703 checks:

```python
if _var.parent is not None:
    _parent_var = self._get_root_parent(_var)
    if _parent_var not in call_arg_map:
        _dtype = _parent_var.type.dtype
        if _dtype in call_arg_dtype_map:
            new_kwargs.append((call_arg_dtype_map[_dtype].name, _parent_var))
```

The problem is that `call_arg_dtype_map` maps derived-type dtypes to the
*first* argument of that type found in the callee. For routines like
`SIGAM_GP` that receive both `YDCPG_BNDS` (type `CPG_BNDS_TYPE`) and
`YDCPG_OPTS` (type `CPG_OPTS_TYPE`), the dtype map finds `YDCPG_BNDS` for
`bnds%kidia` but NOT `YDCPG_OPTS` — because `YDCPG_OPTS` is a different
type and its components (like `YDCPG_OPTS%KLON`) may not appear in
`relevant_vars` at all (if the driver doesn't assign to them in the loop
header).

Additionally, `new_kwargs = set(new_kwargs)` (line 699) gives
**nondeterministic ordering** because Python sets don't preserve insertion
order for tuples. This causes cosmetic kwarg ordering differences even when
the kwargs are correct.

**Secondary issue**: `lassie` also misses the `local_YDCPG_BNDS` assignment
blocks because `CreateLocalCopiesTransformation` gates on
`LowerBlockIndex` trafo_data — if the trafo_data didn't propagate
`YDCPG_BNDS` properly to `lassie`, the local copy isn't created.

### Source code reference
- `block_index_transformations.py:640-703` — `process_kernel` relevant_vars
  collection and kwarg generation
- `block_index_transformations.py:856-891` — `process_driver` relevant_vars
  (same logic)
- `block_index_transformations.py:699` — `new_kwargs = set(new_kwargs)`
  (nondeterministic)

### Proposed fix

**Fix 1 (kwargs completeness):** When building `new_kwargs`, if `_parent_var`
(e.g. `YDCPG_BNDS`) is already in `call_arg_map`, it's skipped via
`already_arg`. But `YDCPG_OPTS` may never appear in `relevant_vars` because
the driver doesn't assign to `opts%klon` in the loop header — it was already
set before the loop. The fix is to also include variables used in expressions
within the loop body (not just header assignments), or to have a broader
scan that identifies ALL derived-type parents of `block_dim.indices`-related
variables.

**Fix 2 (ordering):** Replace `new_kwargs = set(new_kwargs)` with an
`OrderedSet` or `dict.fromkeys()` pattern to preserve deterministic ordering.

### Classification
- **Loki code change** (block_index_transformations.py)

### Test sketch

```python
def test_cat9_missing_kwargs_bnds_opts():
    """
    When a kernel calls a sub-kernel that expects both BNDS and OPTS
    arguments, LowerBlockIndexTransformation.process_kernel must add
    BOTH as keyword arguments to the call.
    """
    # 1. Create driver → kernel → sub_kernel
    # 2. Driver loop header has: bnds%kidia = ..., bnds%kfdia = ...
    # 3. sub_kernel expects (bnds, opts, ...) — two derived-type args
    # 4. Apply LowerBlockIndexTransformation
    # 5. Assert sub_kernel call has BOTH bnds AND opts kwargs
```

```python
def test_cat9_deterministic_kwarg_ordering():
    """
    Keyword arguments added by LowerBlockIndexTransformation must
    have deterministic ordering (not dependent on set iteration order).
    """
    # Run the transformation multiple times
    # Assert kwarg order is identical each time
```

---

## Category 10 — Derived-type components in `!$acc parallel loop gang private()`

### Affected files
- `ecphys_setup_layer` — `private(YDGEOMETRY%YRGEM, YDGEOMETRY%YRDIM, YYTXYB)`
  → should have no private clause (or only `YYTXYB`)

### Root cause

Two bugs in `annotate_driver_loop` (`annotate.py:331`):

**Bug 1 — Dead parameter:**
The method signature is:
```python
def annotate_driver_loop(self, loop, acc_vars, privatise_derived_types=True):
```
But line 353 uses:
```python
if privatise_derived_types:  # <- should be the parameter, but it's correct here
```
Wait — actually the code at line 353 does use the parameter name
`privatise_derived_types`. However, when called from `transform_subroutine`
for `role == 'kernel'` (line 188), it passes `privatise_derived_types=False`.
When called for `role == 'driver'` (line 205), it does NOT pass the parameter
at all, so the default `True` is used.

The actual issue for `ecphys_setup_layer` is that this routine has
`role == 'kernel'` but contains driver loops (because it has calls to target
routines inside a loop). When `annotate_driver_loop` is called with
`privatise_derived_types=False`, the derived-type structs like
`YDGEOMETRY%YRGEM` should NOT be privatised. **But they still are.**

Looking more carefully: when `privatise_derived_types` is `True` (the default
for drivers), the code at line 357-371 collects structs and filters them.
The filter at line 363-370 (`_is_non_local_root`) is supposed to exclude
derived-type components whose root variable has intent or is imported. But
for `YDGEOMETRY`, which IS a subroutine argument (has `intent(in)`), the
filter should exclude `YDGEOMETRY%YRGEM` and `YDGEOMETRY%YRDIM`.

**The real bug**: The `_is_non_local_root` filter was added by us (commit
`57022668` or similar) but only runs when `privatise_derived_types=True`. For
`ecphys_setup_layer`, where `privatise_derived_types=False` is passed, the
struct privatisation block (lines 353-384) is completely skipped. BUT the
`arrays` list (line 346-350) still includes derived-type array components
like `YDGEOMETRY%YRDIM` because they pass the `isinstance(v, sym.Array)`
check.

So the problem is actually that `arrays` (line 346) includes derived-type
component arrays, and these are NOT filtered by the `_is_non_local_root`
logic (which is inside the `if privatise_derived_types:` block). The
`private_sym` list ends up containing `YDGEOMETRY%YRGEM` etc.

**Additionally**: even when `privatise_derived_types=True` (for real
drivers), the `_is_non_local_root` filter needs to be robust. Currently it
only checks `root_type.intent` and `root_type.imported`, but in some cases
the scope resolution may fail silently.

### Source code reference
- `annotate.py:331-407` — `annotate_driver_loop`
- `annotate.py:346-350` — `arrays` filtering (doesn't exclude arg components)
- `annotate.py:353-384` — `privatise_derived_types` block
- `annotate.py:188` — kernel calls with `privatise_derived_types=False`

### Proposed fix

**Fix 1**: Add the `_is_non_local_root` filter to the `arrays` collection
as well (not just the `structs` block). Array components whose root variable
is a subroutine argument or imported should never be privatised:

```python
arrays = [v for v in arrays if not _is_non_local_root(v)]
```

This should be applied regardless of `privatise_derived_types`.

**Fix 2**: Move the `_is_non_local_root` function to a shared location
(class method or module-level) so it can be used in both the arrays and
structs filtering paths.

### Classification
- **Loki code change** (annotate.py)

### Test sketch

```python
def test_cat10_no_arg_components_in_private():
    """
    Derived-type components of subroutine arguments (e.g. YDGEOMETRY%YRDIM)
    must NOT appear in the !$acc parallel loop gang private() clause.
    Only truly local variables should be privatised.
    """
    # 1. Create a kernel that has a driver loop internally
    # 2. The kernel receives YDGEOMETRY (intent(in)) as argument
    # 3. Inside the driver loop, YDGEOMETRY%YRDIM and YDGEOMETRY%YRGEM are used
    # 4. Apply SCCAnnotateTransformation with privatise_derived_types=False
    # 5. Assert private() clause does NOT contain YDGEOMETRY%YRDIM or YDGEOMETRY%YRGEM
    # 6. Assert private() clause DOES contain any truly local derived-types
```

---

## Category 11 — Nested block loop for calls outside main block loop

### Affected files
- `lacdyn` — `CALL LAVABO_LOKI(...)` needs its own `!$acc parallel loop gang`
  wrapper, but doesn't get one

### Root cause

`SCCBlockSectionTransformation.extract_block_sections` (line 212+) splits the
routine body into subsections at `!$loki small-kernels` separator nodes, then
filters subsections for those referencing `block_dim.indices`:

```python
subsections = [s for s in subsections
    if any([index in list(FindVariables().visit(s))
            for index in block_dim.indices])]
```

The `LAVABO` call's subsection doesn't reference any `block_dim.indices`
variables yet — those variables (like `IBL`) are only added BY the block
loop wrapper. This is a **chicken-and-egg problem**: the subsection is
filtered out because it doesn't have `IBL`, but it would have `IBL` if it
weren't filtered out.

In the IFS code, `lacdyn` has multiple `!$loki small-kernels` pragmas. The
main block section contains most calls. But `CALL LAVABO(...)` is outside
the main section — it's in a separate code region after the main block loop.
When `extract_block_sections` runs, LAVABO's section doesn't have `IBL`
references, so it's dropped. The result is that LAVABO's `_LOKI` call
appears in the generated code without a block loop wrapper, meaning it
executes only once instead of per-block.

### Source code reference
- `block.py:211-278` — `extract_block_sections`
- `block.py:254` — the `block_dim.indices` filter that drops LAVABO's section
- `block.py:281-299` — `get_trimmed_sections`

### Proposed fix

**Option A (IFS source change):** Add `!$loki small-kernels` pragma before
the LAVABO call in `lacdyn.F90`. This would make LAVABO a separator node,
and its section would be processed. Simple and targeted.

**Option B (Loki code change):** Modify the subsection filter to also include
sections that contain calls to routines that will need block indices (i.e.,
routines whose names match `_LOKI` suffix or that are in the targets list).
This addresses the chicken-and-egg problem:

```python
subsections = [s for s in subsections
    if any([index in FindVariables().visit(s) for index in block_dim.indices])
    or any(call.name.endswith('_LOKI') for call in FindNodes(CallStatement).visit(s))
]
```

**Option C (Loki code change — post-processing):** After all block sections
have been processed, scan for orphaned `_LOKI` calls (calls outside any
block loop) and wrap them in their own block loop.

### Classification
- **IFS source change** (Option A) is simplest and most pragmatic
- **Loki code change** (Option B or C) for a generic solution

### Test sketch

```python
def test_cat11_call_outside_main_section_gets_block_loop():
    """
    When a kernel has multiple !$loki small-kernels sections and a call
    to a target routine is outside all of them, it must still get wrapped
    in its own block loop.
    """
    # 1. Create a kernel with two code sections:
    #    Section 1: main computation with !$loki small-kernels + CALL kernel_a(...)
    #    Section 2: CALL kernel_b(...) — no pragma, outside section 1
    # 2. Apply SCCBlockSectionTransformation + SCCBlockSectionToLoopTransformation
    # 3. Assert kernel_b's call IS inside a block loop (DO IBL=1,NGPBLKS)
    # 4. Assert kernel_a's call is also inside a block loop (normal case)
```

---

## Category 12 — Stack/pool allocator generated for kernel routines

### Affected files
- `larcinb` — Working version has ISTSZ/ZSTACK/ALLOCATE/DEALLOCATE commented
  out; Loki output generates them actively

### Root cause

`TemporariesPoolAllocatorPerDrvLoopTransformation.transform_subroutine`
calls `find_driver_loops(routine.body, targets)` for every routine regardless
of role. In `larcinb` (role = `'kernel'`), there is a
`DO JFLD = 1, SIZE(YDVARS%GFL_PTR)` loop that contains calls to target
routines. `find_driver_loops` returns this as a "driver loop" because it
matches the heuristic (loop containing target calls).

Once a "driver loop" is found, the pool allocator generates the full
ISTSZ/ZSTACK/ALLOCATE/DEALLOCATE setup inside the kernel. This is wrong:
kernels should not have their own pool allocator setup.

### Source code reference
- `pool_allocator_per_drv_loop.py:240-258` — the `if driver_loops:` block
  that generates ISTSZ/ZSTACK unconditionally
- `pool_allocator_per_drv_loop.py:247-248` — both `role == 'kernel'` and
  `role == 'driver'` take the same path

### Proposed fix

**Guard the per-driver-loop setup with a role check:**

```python
if driver_loops:
    if role == 'driver':
        self.add_driver_imports(routine)
        drv_loop_map = {}
        for drv_loop in driver_loops:
            stack_size = self._determine_stack_size(...)
            drv_loop_map[drv_loop] = self.create_pool_allocator_drv_loop(...)
        # ... etc
    elif role == 'kernel':
        # For kernels, only compute aggregate stack size for propagation
        # Do NOT create ISTSZ/ZSTACK variables
        for drv_loop in driver_loops:
            stack_size = self._determine_stack_size(...)
            # Store for aggregation but don't generate setup code
```

This is straightforward: kernels should never generate their own ISTSZ/ZSTACK
setup. The driver is responsible for allocating the scratch space; the kernel
only receives the stack pointer as an argument.

### Classification
- **Loki code change** (pool_allocator_per_drv_loop.py)

### Test sketch

```python
def test_cat12_kernel_no_pool_allocator_setup():
    """
    When a kernel contains a loop with calls to target routines,
    the pool allocator must NOT generate ISTSZ/ZSTACK/ALLOCATE/DEALLOCATE
    at the kernel level.
    """
    # 1. Create a kernel with: DO JFLD=1,N ... CALL sub_kernel(...) ... END DO
    # 2. sub_kernel has temporaries that need pool allocation
    # 3. Apply the full pipeline
    # 4. Assert kernel does NOT have 'ISTSZ' or 'ZSTACK' in its variables
    # 5. Assert kernel does NOT have ALLOCATE/DEALLOCATE statements
    # 6. Assert kernel's sub_kernel call DOES have YDSTACK_L/YDSTACK_U kwargs
```

---

## Summary Table

| Cat | Description | Root cause location | Fix type | Complexity |
|-----|-------------|-------------------|----------|------------|
| 3 | Missing ISTSZ | pool_allocator_per_drv_loop.py | Loki | Medium |
| 5 | Rank mismatch VERDISINT | block_index_transformations.py | IFS source preferred | High |
| 6 | Missing IBL in host path | block_index_transformations.py | Loki | Medium |
| 9 | Missing kwargs BNDS/OPTS | block_index_transformations.py | Loki | Medium |
| 10 | Derived-type in private() | annotate.py | Loki | Low |
| 11 | Call outside block section | block.py | IFS source or Loki | Low-Medium |
| 12 | Pool allocator in kernel | pool_allocator_per_drv_loop.py | Loki | Low |

### Recommended fix order (by complexity, dependencies)
1. **Cat 12** (Low) — Simple role guard in pool allocator
2. **Cat 10** (Low) — Add non-local-root filter to arrays list
3. **Cat 3** (Medium) — Closely related to Cat 12; fix together
4. **Cat 9** (Medium) — Fix kwargs collection logic + ordering
5. **Cat 6** (Medium) — Second pass for host-path calls
6. **Cat 11** (Low-Medium) — IFS source change or filter fix
7. **Cat 5** (High) — Rank mismatch; likely needs IFS source change

### Potential "hacky temporary solutions"
- **Cat 11**: Add `!$loki small-kernels` pragma to LAVABO call in IFS source
  (simple but not generic)
- **Cat 5**: Skip block-index injection for mismatched-rank arguments and
  emit a warning (safe but incomplete)
- **Cat 6**: Hard-code detection of `_LOKI` suffix variant to find the
  paired non-`_LOKI` call (fragile)

### Potential "better as IFS source change"
- **Cat 5**: Change `EDOT` field declaration to avoid sequence association
- **Cat 11**: Add `!$loki small-kernels` pragma to LAVABO call site
- **Cat 3 (lapineb)**: Restructure driver to keep block loop for pool
  allocator purposes
