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

from loki.batch import Transformation
from loki.logging import debug

class CMakePlanTransformation(Transformation):
    """
    Generate the "plan file" for CMake

    The plan file is a CMake file defining three lists:

    * ``LOKI_SOURCES_TO_TRANSFORM``: The list of files that are
        processed in the dependency graph
    * ``LOKI_SOURCES_TO_APPEND``: The list of files that are created
        and have to be added to the build target as part of the processing
    * ``LOKI_SOURCES_TO_REMOVE``: The list of files that are no longer
        required (because they have been replaced by transformed files) and
        should be removed from the build target.

    Parameters
    ----------
    rootpath : str (optional)

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
        if self.rootpath is not None:
            sourcepath = sourcepath.relative_to(self.rootpath)

        newsource = item.trafo_data['FileWriteTransformation']['path']

        debug(f'Planning:: {item.name} (role={item.role}, mode={item.mode})')

        if newsource not in self.sources_to_append:
            self.sources_to_transform += [sourcepath]
            if item.replicate:
                # Add new source file next to the old one
                self.sources_to_append += [newsource]
            else:
                # Replace old source file to avoid ghosting
                self.sources_to_append += [newsource]
                self.sources_to_remove += [sourcepath]

    def write_plan(self, filepath):
        with Path(filepath).open('w') as f:
            s_transform = '\n'.join(f'    {s}' for s in self.sources_to_transform)
            f.write(f'set( LOKI_SOURCES_TO_TRANSFORM \n{s_transform}\n   )\n')

            s_append = '\n'.join(f'    {s}' for s in self.sources_to_append)
            f.write(f'set( LOKI_SOURCES_TO_APPEND \n{s_append}\n   )\n')

            s_remove = '\n'.join(f'    {s}' for s in self.sources_to_remove)
            f.write(f'set( LOKI_SOURCES_TO_REMOVE \n{s_remove}\n   )\n')
