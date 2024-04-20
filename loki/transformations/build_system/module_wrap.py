# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.batch import Transformation
from loki.expression import Variable
from loki.ir import Import, Section, Interface, FindNodes, Transformer
from loki.module import Module
from loki.subroutine import Subroutine
from loki.tools import as_tuple


__all__ = ['ModuleWrapTransformation']


class ModuleWrapTransformation(Transformation):
    """
    Utility transformation that ensures all transformed kernel
    subroutines are wrapped in a module

    The module name is derived from the subroutine name and :data:`module_suffix`.

    Any previous import of wrapped subroutines via interfaces or C-style header
    imports of interface blocks is replaced by a Fortran import (``USE``).

    Parameters
    ----------
    module_suffix : str
        Special suffix to signal module names like `_MOD`
    replace_ignore_items : bool
        Debug flag to toggle the replacement of calls to subroutines
        in the ``ignore``. Default is ``True``.
    """

    # This transformation is applied over the file graph
    traverse_file_graph = True

    # This transformation recurses from the Sourcefile down
    recurse_to_modules = True
    recurse_to_procedures = True
    recurse_to_internal_procedures = False

    # This transformation changes the names of items and creates new items
    renames_items = True
    creates_items = True

    def __init__(self, module_suffix, replace_ignore_items=True):
        self.module_suffix = module_suffix
        self.replace_ignore_items = replace_ignore_items

    def transform_file(self, sourcefile, **kwargs):
        """
        For kernel routines, wrap each subroutine in the current file in a module
        """
        role = kwargs.get('role')

        if items := kwargs.get('items'):
            # We consider the sourcefile to be a "kernel" file if all items are kernels
            if all(item.role == 'kernel' for item in items):
                role = 'kernel'
            else:
                role = 'driver'

        if role == 'kernel':
            self.module_wrap(sourcefile)

    def transform_module(self, module, **kwargs):
        """
        Update imports of wrapped subroutines
        """
        self.update_imports(module, imports=module.imports, **kwargs)

    def transform_subroutine(self, routine, **kwargs):
        """
        Update imports of wrapped subroutines
        """
        if item := kwargs.get('item'):
            # Rename the item if it has suddenly a parent
            if routine.parent and routine.parent.name.lower() != item.scope_name:
                item.name = f'{routine.parent.name.lower()}#{item.local_name}'

        # Note, C-style imports can be in the body, so use whole IR
        imports = FindNodes(Import).visit(routine.ir)
        self.update_imports(routine, imports=imports, **kwargs)

        # Interface blocks can only be in the spec
        intfs = FindNodes(Interface).visit(routine.spec)
        self.replace_interfaces(routine, intfs=intfs, **kwargs)

    def module_wrap(self, sourcefile):
        """
        Wrap target subroutines in modules and replace in source file.
        """
        for routine in sourcefile.subroutines:
            # Create wrapper module and insert into file, replacing the old
            # standalone routine
            modname = f'{routine.name}{self.module_suffix}'
            module = Module(name=modname, contains=Section(body=as_tuple(routine)))
            routine.parent = module
            sourcefile.ir._update(body=as_tuple(
                module if c is routine else c for c in sourcefile.ir.body
            ))

    def update_imports(self, source, imports, **kwargs):
        """
        Update imports of wrapped subroutines.
        """
        from loki.batch import SchedulerConfig  # pylint: disable=import-outside-toplevel,cyclic-import

        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets')))
        if self.replace_ignore_items and (item := kwargs.get('item')):
            targets += tuple(str(i).lower() for i in item.ignore)

        def _update_item(proc_name, module_name):
            if item and (matched_keys := SchedulerConfig.match_item_keys(proc_name, item.ignore)):
                # Add the module wrapped but ignored items to the block list because we won't be able to
                # find them as dependencies under their new name anymore
                item.config['block'] = as_tuple(item.block) + tuple(
                    module_name for name in item.ignore if name in matched_keys
                )

        # Transformer map to remove any outdated imports
        removal_map = {}

        # We go through the IR, as C-imports can be attributed to the body
        for im in imports:
            if im.c_import:
                target_symbol, *suffixes = im.module.lower().split('.', maxsplit=1)
                if targets and target_symbol.lower() in targets and not 'func.h' in suffixes:
                    # Create a new module import with explicitly qualified symbol
                    modname = f'{target_symbol}{self.module_suffix}'
                    _update_item(target_symbol.lower(), modname)
                    new_symbol = Variable(name=target_symbol, scope=source)
                    new_import = im.clone(module=modname, c_import=False, symbols=(new_symbol,))
                    source.spec.prepend(new_import)

                    # Mark current import for removal
                    removal_map[im] = None

        # Apply any scheduled import removals to spec and body
        if removal_map:
            source.spec = Transformer(removal_map).visit(source.spec)
            if isinstance(source, Subroutine):
                source.body = Transformer(removal_map).visit(source.body)

    def replace_interfaces(self, source, intfs, **kwargs):
        """
        Update explicit interfaces to actively transformed subroutines.
        """
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets')))
        if self.replace_ignore_items and (item := kwargs.get('item')):
            targets += tuple(str(i).lower() for i in item.ignore)

        # Transformer map to remove any outdated interfaces
        removal_map = {}

        for i in intfs:
            for b in i.body:
                if isinstance(b, Subroutine):
                    if targets and b.name.lower() in targets:
                        # Create a new module import with explicitly qualified symbol
                        modname = f'{b.name}{self.module_suffix}'
                        new_symbol = Variable(name=f'{b.name}', scope=source)
                        new_import = Import(module=modname, c_import=False, symbols=(new_symbol,))
                        source.spec.prepend(new_import)

                        # Mark current import for removal
                        removal_map[i] = None

        # Apply any scheduled interface removals to spec
        if removal_map:
            source.spec = Transformer(removal_map).visit(source.spec)
