"""
Transformations to be used in build-system level tasks
"""

from pathlib import Path

from loki.logging import info
from loki.transform.transformation import Transformation

__all__ = ['CMakePlanner']


class CMakePlanner(Transformation):
    """
    Generates a list of files to add, remove or replace in a CMake
    target's list of sources

    This is intended to be used in a :any:`Scheduler` traversal triggered
    during the configuration phase of a build (e.g., using ``execute_process``)
    to generate a CMake plan file (using :meth:`write_planfile`).
    This file set variables ``LOKI_SOURCES_TO_TRANSFORM``,
    ``LOKI_SOURCES_TO_APPEND``, and ``LOKI_SOURCES_TO_REMOVE`` that can then
    be used to update a target's ``SOURCES`` property via
    ``get_target_property`` and ``set_property``.

    Attributes
    ----------
    sources_to_append : list of str
        Newly generated source files that need to be added to the target
    sources_to_remove : list of str
        The source files that are replaced and must be removed from the target
    sources_to_transform : list of str
        The source files that are going to be transformed by Loki
        transformations

    Parameters
    ----------
    rootpath : :any:`pathlib.Path` or str
        The base directory of the source tree
    mode : str
        The name of the transformation mode (which is going to be inserted
        into the file name of new source files)
    build : :any:`pathlib.Path` or str
        The target directory for generate source files
    """

    def __init__(self, rootpath, mode, build=None):
        self.build = None if build is None else Path(build)
        self.mode = mode

        self.rootpath = Path(rootpath).resolve()
        self.sources_to_append = []
        self.sources_to_remove = []
        self.sources_to_transform = []

    def transform_subroutine(self, routine, **kwargs):
        """
        Insert the current subroutine into the lists of source files to
        process, add and remove, if part of the plan

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine object to process
        item : :any:`Item`
            The corresponding work item from the :any:`Scheduler`
        role : str
            The routine's role
        """
        item = kwargs['item']
        role = kwargs.get('role')

        # Back out, if this is Subroutine is not part of the plan
        if item.name.lower() != routine.name.lower():
            return

        sourcepath = item.path.resolve()
        newsource = sourcepath.with_suffix(f'.{self.mode.lower()}.F90')
        if self.build is not None:
            newsource = self.build/newsource.name

        # Make new CMake paths relative to source again
        sourcepath = sourcepath.relative_to(self.rootpath)

        info(f'Planning:: {routine.name} (role={role}, mode={self.mode})')

        self.sources_to_transform += [sourcepath]

        # Inject new object into the final binary libs
        if item.replicate:
            # Add new source file next to the old one
            self.sources_to_append += [newsource]
        else:
            # Replace old source file to avoid ghosting
            self.sources_to_append += [newsource]
            self.sources_to_remove += [sourcepath]

    def write_planfile(self, filepath):
        """
        Write the CMake plan file at :data:`filepath`
        """
        info(f'[Loki] CMakePlanner writing plan: {filepath}')
        with Path(filepath).open('w') as f:
            s_transform = '\n'.join(f'    {s}' for s in self.sources_to_transform)
            f.write(f'set( LOKI_SOURCES_TO_TRANSFORM \n{s_transform}\n   )\n')

            s_append = '\n'.join(f'    {s}' for s in self.sources_to_append)
            f.write(f'set( LOKI_SOURCES_TO_APPEND \n{s_append}\n   )\n')

            s_remove = '\n'.join(f'    {s}' for s in self.sources_to_remove)
            f.write(f'set( LOKI_SOURCES_TO_REMOVE \n{s_remove}\n   )\n')
