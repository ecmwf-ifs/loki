# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Transformations to be used for exposing planned changes to
the build system
"""

from pathlib import Path

from loki.batch.transformation import Transformation
from loki.logging import debug

class CMakePlanTransformation(Transformation):
    """
    Gather the planning information from all :any:`Item.trafo_data` to which
    this information is applied and allows writing a CMake plan file

    This requires that :any:`FileWriteTransformation` has been applied in planning
    mode first.

    Applying this transformation to a :any:`Item` updates internal lists:

    * :attr:`sources_to_transform`: The path of all source files that contain
      objects that are transformed by a Loki transformation in the pipeline
    * :attr:`sources_to_append`: The path of any new source files that exist
      as a consequence of the Loki transformation pipeline, e.g., transformed
      source files that are written.
    * :attr:`sources_to_remove`: The path of any existing source files that
      are to be removed from the compilation target. This includes all items
      that don't have the :any:`Item.replicate` property.

    The :any:`Sourcefile.path` is used to determine the file path from which a
    Fortran sourcefile was read. New paths are provided in
    ``item.trafo_data['FileWriteTransformation']['path']``.

    The method :meth:`write_plan` allows to write the gathered information to
    a CMake file that can be included in the CMake scripts that build a library.
    The plan file is a CMake file defining three lists matching the above:

    * ``LOKI_SOURCES_TO_TRANSFORM``: The list of files that are
        processed in the dependency graph
    * ``LOKI_SOURCES_TO_APPEND``: The list of files that are created
        and have to be added to the build target as part of the processing
    * ``LOKI_SOURCES_TO_REMOVE``: The list of files that are no longer
        required (because they have been replaced by transformed files) and
        should be removed from the build target.

    These lists are used by the Loki CMake wrappers (particularly
    ``loki_transform_target``) to schedule the source updates and update the
    source lists of the CMake target object accordingly.

    Parameters
    ----------
    rootpath : str (optional)
        If given, all paths will be resolved relative to this root directory
    """

    # This transformation is applied over the file graph
    traverse_file_graph = True

    item_filter = None

    def __init__(self, rootpath=None):
        self.rootpath = None if rootpath is None else Path(rootpath).resolve()
        self.sources_to_append = []
        self.sources_to_remove = []
        self.sources_to_transform = []

    def plan_file(self, sourcefile, **kwargs):
        item = kwargs.get('item')
        if not item:
            raise ValueError('No Item provided; required to determine CMake plan')

        if not 'FileWriteTransformation' in item.trafo_data:
            return

        sourcepath = item.path.resolve()

        # This makes sure the sourcepath does in fact exist. Combined with
        # item duplication or other transformations we might end up adding
        # items on-the-fly that did not exist before, with fake paths.
        # There is possibly a better way of doing this, though.
        source_exists = sourcepath.exists()

        if self.rootpath is not None:
            sourcepath = sourcepath.relative_to(self.rootpath)

        newsource = item.trafo_data['FileWriteTransformation']['path']

        debug(f'Planning:: {item.name} (role={item.role}, mode={item.mode})')

        if newsource not in self.sources_to_append:
            if source_exists:
                self.sources_to_transform += [sourcepath]
            if item.replicate:
                # Add new source file next to the old one
                self.sources_to_append += [newsource]
            else:
                # Replace old source file to avoid ghosting
                self.sources_to_append += [newsource]
                if source_exists:
                    self.sources_to_remove += [sourcepath]

    def write_plan(self, filepath):
        """
        Write the CMake plan file to :data:`filepath`
        """
        with Path(filepath).open('w') as f:
            s_transform = '\n'.join(f'    {s}' for s in self.sources_to_transform)
            f.write(f'set( LOKI_SOURCES_TO_TRANSFORM \n{s_transform}\n   )\n')

            s_append = '\n'.join(f'    {s}' for s in self.sources_to_append)
            f.write(f'set( LOKI_SOURCES_TO_APPEND \n{s_append}\n   )\n')

            s_remove = '\n'.join(f'    {s}' for s in self.sources_to_remove)
            f.write(f'set( LOKI_SOURCES_TO_REMOVE \n{s_remove}\n   )\n')
