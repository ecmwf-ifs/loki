==================
Programming models
==================

Loki directives
---------------

Loki uses an internal set of directives as an intermediate annotation for data movement
and parallelisation concepts. Transformations, such as the :any:`SCCAnnotateTransformation`,
insert these directives, or they can be written into the original Fortran source code.
The :any:`PragmaModelTransformation` should be used, as one of the final steps in a processing
pipeline, to translate these directives to the corresponding instructions for the chosen
programming model.

Currently, Loki supports OpenACC and some OpenMP. The following table gives a summary of how
Loki directives are translated to the corresponding pragmas in either programming model:

.. csv-table:: Loki generic pragmas to pragma model mapping
   :file: /loki_pragma_model.csv
   :widths: 100, 100, 100
   :header-rows: 1

Collapsing SCC-seq driver loops
------------------------------

The SCC-seq pipelines accept ``collapse_gang_vector=True`` to combine an
already perfectly nested gang/vector loop pair using ``collapse(2)``. The
option is disabled by default and supports ``directive='openacc'``,
``directive='omp-gpu'`` and generic Loki output (``directive=False``). CPU
OpenMP configurations retain the original separate loop annotations.

:any:`SCCCollapseTransformation` runs after temporary-storage synthesis and
before :any:`PragmaModelTransformation`. It requires rectangular, invariant
loop bounds and unit steps, and accepts ordinary comments between the loops.
It preserves ``vlength`` and ``async`` clauses. Nests with private/firstprivate
or reduction clauses, conflicting annotations, unsupported control flow, or
executable statements between the loops are left unchanged, with an informational
diagnostic. In particular, stack-pointer initialization between the loops is
not moved. Existing SCC annotations provide the underlying parallelism contract;
this pass checks whether their loop mappings can be combined.

The resulting generic outer directive is ``!$loki loop gang vector collapse(2)``;
the inner vector directive is removed and both Fortran loops remain present.
OpenACC lowering produces ``!$acc parallel loop gang vector collapse(2)``.
OpenMP GPU lowering produces
``!$omp target teams distribute parallel do collapse(2) thread_limit(...)``,
with ``vlength`` supplying ``thread_limit`` when present. The matching generic
end directive is ``!$loki end loop gang vector``, which lowers to
``!$omp end target teams distribute parallel do``. As with existing OpenMP GPU
loop mappings, ``async`` is not translated and execution is synchronous.
Combined gang/vector directives cannot currently be lowered to CPU OpenMP.
