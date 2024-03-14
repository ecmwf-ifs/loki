# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from loki.transform.transformation import Transformation
from loki.subroutine import Subroutine
from loki.module import Module
from loki.visitors import FindNodes, Transformer
from loki.ir import CallStatement, Import, Section, Interface
from loki.expression import Variable, FindInlineCalls, SubstituteExpressions
from loki.backend import fgen
from loki.tools import as_tuple


__all__ = ['DependencyTransformation', 'ModuleWrapTransformation']


class DependencyTransformation(Transformation):
    """
    Basic :any:`Transformation` class that facilitates dependency
    injection for transformed :any:`Module` and :any:`Subroutine`
    into complex source trees

    This transformation appends a provided ``suffix`` argument to
    transformed subroutine and module objects and changes the target
    names of :any:`Import` and :any:`CallStatement` nodes on the call-site
    accordingly.

    For subroutines declared via an interface block, these interfaces
    are updated accordingly. For subroutines that are not wrapped in a
    module, an updated interface block is also written as a header file
    to :data:`include_path`. Where interface blocks to renamed subroutines
    are included via C-style imports, the import name is updated accordingly.

    To ensure that every subroutine is wrapped in a module, the
    accompanying :any:`ModuleWrapTransformation` should be applied
    first. This restores the behaviour of the ``module`` mode in an earlier
    version of this transformation.

    When applying the transformation to a source object, one of two
    "roles" can be specified via the ``role`` keyword:

    * ``'driver'``: Only renames imports and calls to kernel routines
    * ``'kernel'``: Renames routine or enclosing modules, as well as
      renaming any further imports and calls.

    Note that ``routine.apply(transformation, role='driver')`` entails
    that the ``routine`` still mimicks its original counterpart and
    can therefore be used as a drop-in replacement during compilation
    that then diverts the dependency tree to the modified sub-tree.

    Parameters
    ----------
    suffix : str
        The suffix to apply during renaming
    module_suffix : str
        Special suffix to signal module names like `_MOD`
    include path : path
        Directory for generating additional header files
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

    def __init__(self, suffix, module_suffix=None, include_path=None, replace_ignore_items=True):
        self.suffix = suffix
        self.module_suffix = module_suffix
        self.replace_ignore_items = replace_ignore_items
        self.include_path = None if include_path is None else Path(include_path)

    def transform_module(self, module, **kwargs):
        """
        Rename kernel modules and re-point module-level imports.
        """
        if kwargs.get('role') == 'kernel':
            # Change the name of kernel modules
            module.name = self.derive_module_name(module.name)

        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets')))
        if self.replace_ignore_items and (item := kwargs.get('item')):
            targets += tuple(str(i).lower() for i in item.ignore)
        self.rename_imports(module, imports=module.imports, targets=targets)

    def transform_subroutine(self, routine, **kwargs):
        """
        Rename kernel subroutine and all imports and calls to target routines

        For subroutines that are not wrapped in a module, re-generate the interface
        block.
        """
        role = kwargs.get('role')
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets')))
        if self.replace_ignore_items and (item := kwargs.get('item')):
            targets += tuple(str(i).lower() for i in item.ignore)

        if role == 'kernel':
            if routine.name.endswith(self.suffix):
                # This is to ensure that the transformation is idempotent if
                # applied more than once to a routine
                return
            # Change the name of kernel routines
            if routine.is_function and not routine.result_name:
                self.update_result_var(routine)
            routine.name += self.suffix

        self.rename_calls(routine, targets=targets)

        # Note, C-style imports can be in the body, so use whole IR
        imports = FindNodes(Import).visit(routine.ir)
        self.rename_imports(routine, imports=imports, targets=targets)

        # Interface blocks can only be in the spec
        intfs = FindNodes(Interface).visit(routine.spec)
        self.rename_interfaces(intfs, targets=targets)

        if role == 'kernel' and not routine.parent:
            # Re-generate C-style interface header
            self.generate_interfaces(routine)

    def derive_module_name(self, modname):
        """
        Utility to derive a new module name from :attr:`suffix` and :attr:`module_suffix`

        Parameters
        ----------
        modname : str
            Current module name
        """

        # First step through known suffix variants to determine canonical basename
        if self.module_suffix and modname.lower().endswith(self.module_suffix.lower()):
            # Remove the module_suffix, if present
            idx = modname.lower().rindex(self.module_suffix.lower())
            modname = modname[:idx]
        if modname.lower().endswith(self.suffix.lower()):
            # Remove the dependency injection suffix, if present
            idx = modname.lower().rindex(self.suffix.lower())
            modname = modname[:idx]

        # Suffix combination to canonical basename
        if self.module_suffix:
            return f'{modname}{self.suffix}{self.module_suffix}'
        return f'{modname}{self.suffix}'

    def update_result_var(self, routine):
        """
        Update name of result variable for function calls.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The function object for which the result variable is to be renamed
        """
        assert routine.name in routine.variables

        vmap = {
            v: v.clone(name=v.name + self.suffix)
            for v in routine.variables if v == routine.name
        }
        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)

    def rename_calls(self, routine, targets=None):
        """
        Update :any:`CallStatement` and :any:`InlineCall` to actively
        transformed procedures

        Parameters
        ----------
        targets : list of str
            Optional list of subroutine names for which to modify the corresponding
            calls. If not provided, all calls are updated
        """
        members = [r.name for r in routine.subroutines]

        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in members:
                continue
            if targets is None or call.name in targets:
                call._update(name=call.name.clone(name=f'{call.name}{self.suffix}'))

        for call in FindInlineCalls(unique=False).visit(routine.body):
            if call.function in members:
                continue
            if targets is None or call.function in targets:
                call.function = call.function.clone(name=f'{call.name}{self.suffix}')

    def rename_imports(self, source, imports, targets=None):
        """
        Update imports of actively transformed subroutines.

        Parameters
        ----------
        source : :any:`ProgramUnit`
            The IR object to transform
        imports : list of :any:`Import`
            The list of imports to update. This includes both, C-style header includes
            and Fortran import statements (``USE`` and ``IMPORT``)
        targets : list of str
            Optional list of subroutine names for which to modify imports
        """
        # We don't want to rename module variable imports, so we build
        # a list of calls to further filter the targets
        if isinstance(source, Module):
            calls = ()
            for routine in source.subroutines:
                calls += tuple(str(c.name).lower() for c in FindNodes(CallStatement).visit(routine.body))
                calls += tuple(str(c.name).lower() for c in FindInlineCalls().visit(routine.body))
        else:
            calls = tuple(str(c.name).lower() for c in FindNodes(CallStatement).visit(source.body))
            calls += tuple(str(c.name).lower() for c in FindInlineCalls().visit(source.body))

        # Import statements still point to unmodified call names
        calls = [call.replace(f'{self.suffix.lower()}', '') for call in calls]

        # We go through the IR, as C-imports can be attributed to the body
        for im in imports:
            if im.c_import:
                target_symbol = im.module.split('.')[0].lower()
                if targets and target_symbol.lower() in targets:
                    # Modify the the basename of the C-style header import
                    s = '.'.join(im.module.split('.')[1:])
                    im._update(module=f'{target_symbol}{self.suffix}.{s}')

            else:
                # Modify module import if it imports any targets
                if targets and any(s in targets and s in calls for s in im.symbols):
                    # Append suffix to all target symbols
                    symbols = as_tuple(s.clone(name=f'{s.name}{self.suffix}')
                                       if s in targets else s for s in im.symbols)
                    module_name = self.derive_module_name(im.module)
                    im._update(module=module_name, symbols=symbols)

                # TODO: Deal with unqualified blanket imports

    def rename_interfaces(self, intfs, targets=None):
        """
        Update explicit interfaces to actively transformed subroutines.

        Parameters
        ----------
        intfs : list of :any:`Interface`
            The list of interfaces to update.
        targets : list of str
            Optional list of subroutine names for which to modify interfaces
        """
        for i in intfs:
            for routine in i.body:
                if isinstance(routine, Subroutine):
                    if targets and routine.name.lower() in targets:
                        routine.name = f'{routine.name}{self.suffix}'

    def generate_interfaces(self, routine):
        """
        Generate external header file with interface block for this subroutine.
        """
        print(f"generate_interfaces: {routine}")
        # No need to rename here, as this has already happened before
        intfb_path = self.include_path/f'{routine.name.lower()}.intfb.h'
        with intfb_path.open('w') as f:
            f.write(fgen(routine.interface))


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

    def __init__(self, module_suffix, replace_ignore_items=True):
        self.module_suffix = module_suffix
        self.replace_ignore_items = replace_ignore_items

    def transform_file(self, sourcefile, **kwargs):
        """
        For kernel routines, wrap each subroutine in the current file in a module
        """
        items = kwargs.get('items')
        role = kwargs.pop('role')

        if not role and items:
            # We consider the sourcefile to be a "kernel" file if all items are kernels
            if all(item.role == 'kernel' for item in items):
                role = 'kernel'

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
        targets = tuple(str(t).lower() for t in as_tuple(kwargs.get('targets')))
        if self.replace_ignore_items and (item := kwargs.get('item')):
            targets += tuple(str(i).lower() for i in item.ignore)

        # Transformer map to remove any outdated imports
        removal_map = {}

        # We go through the IR, as C-imports can be attributed to the body
        for im in imports:
            if im.c_import:
                target_symbol = im.module.split('.')[0].lower()
                if targets and target_symbol.lower() in targets:
                    # Create a new module import with explicitly qualified symbol
                    modname = f'{target_symbol}{self.module_suffix}'
                    new_symbol = Variable(name=f'{target_symbol}', scope=source)
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
