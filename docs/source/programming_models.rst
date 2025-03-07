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
