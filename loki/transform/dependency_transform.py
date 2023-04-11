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


__all__ = ['DependencyTransformation']


class DependencyTransformation(Transformation):
    """
    Basic :any:`Transformation` class that facilitates dependency
    injection for transformed :any:`Module` and :any:`Subroutine`
    into complex source trees. It does so by appending a provided
    ``suffix`` argument to transformed subroutine and module objects
    and changing the target names of :any:`Import` and
    :any:`CallStatement` nodes on the call-site accordingly.

    The :any:`DependencyTransformation` provides two ``mode`` options:

    * ``strict`` honors dependencies via C-style headers
    * ``module`` replaces C-style header dependencies with explicit
      module imports

    When applying the transformation to a source object, one of two
    "roles" can be specified via the ``role`` keyword:

    * ``driver``: Only renames imports and calls to kernel routines
    * ``kernel``: Renames routine or enclosing modules, as well as
      renaming any further imports and calls.

    Note that ``routine.apply(transformation, role='driver')`` entails
    that the ``routine`` still mimicks its original counterpart and
    can therefore be used as a drop-in replacement during compilation
    that then diverts the dependency tree to the modified sub-tree.

    Parameters
    ----------
    suffix : str
        The suffix to apply during renaming
    mode : str
        The injection mode to use; either `'strict'` or `'module'`
    module_suffix : str
        Special suffix to signal module names like `_MOD`
    include path : path
        Directory for generating additional header files
    replace_ignore_items : bool
        Debug flag to toggle the replacement of calls to subroutines
        in the ``ignore``. Default is ``True``.
    """

    def __init__(self, suffix, mode='module', module_suffix=None, include_path=None,
                 replace_ignore_items=True):
        self.suffix = suffix
        assert mode in ['strict', 'module']
        self.mode = mode
        self.replace_ignore_items = replace_ignore_items

        self.module_suffix = module_suffix
        self.include_path = None if include_path is None else Path(include_path)

    def transform_subroutine(self, routine, **kwargs):
        """
        Rename driver subroutine and all calls to target routines. In
        'strict' mode, also  re-generate the kernel interface headers.
        """
        role = kwargs.get('role')
        item = kwargs.get('item', None)

        # Bail if this routine is not part of a scheduler traversal
        if item and not item.local_name == routine.name.lower():
            return

        if role == 'kernel':
            # Change the name of kernel routines
            if routine.is_function:
                if not routine.result_name:
                    self.update_result_var(routine)
            routine.name += self.suffix

        self.rename_calls(routine, **kwargs)

        # Note, C-style imports can be in the body, so use whole IR
        imports = FindNodes(Import).visit(routine.ir)
        self.rename_imports(routine, imports=imports, **kwargs)

        # Interface blocks can only be in the spec
        intfs = FindNodes(Interface).visit(routine.spec)
        self.rename_interfaces(routine, intfs=intfs, **kwargs)

        if role == 'kernel' and self.mode == 'strict':
            # Re-generate C-style interface header
            self.generate_interfaces(routine)

    def update_result_var(self, routine):
        """
        Update name of result variable for function calls.
        """

        assert routine.name in routine.variables

        vmap = {}
        for v in routine.variables:
            if v == routine.name:
                vmap.update({v: v.clone(name=v.name + self.suffix)})

        routine.spec = SubstituteExpressions(vmap).visit(routine.spec)
        routine.body = SubstituteExpressions(vmap).visit(routine.body)

    def transform_module(self, module, **kwargs):
        """
        Rename kernel modules and re-point module-level imports.
        """
        role = kwargs.get('role')

        if role == 'kernel':
            # Change the name of kernel modules
            module.name = self.derive_module_name(module.name)

        # Module imports only appear in the spec section
        imports = FindNodes(Import).visit(module.spec)
        self.rename_imports(module, imports=imports, **kwargs)

    def transform_file(self, sourcefile, **kwargs):
        """
        In 'module' mode perform module-wrapping for dependnecy injection.
        """
        role = kwargs.get('role')
        if role == 'kernel' and self.mode == 'module':
            self.module_wrap(sourcefile, **kwargs)

    def rename_calls(self, routine, **kwargs):
        """
        Update calls to actively transformed subroutines.

        :param targets: Optional list of subroutine names for which to
                        modify the corresponding calls.
        """
        targets = as_tuple(kwargs.get('targets', None))
        targets = as_tuple(str(t).upper() for t in targets)
        members = [r.name for r in routine.subroutines]

        if self.replace_ignore_items:
            item = kwargs.get('item', None)
            targets += as_tuple(str(i).upper() for i in item.ignore) if item else ()

        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in members:
                continue
            if targets is None or call.name in targets:
                call._update(name=call.name.clone(name=f'{call.name}{self.suffix}'))

        for call in FindInlineCalls(unique=False).visit(routine.body):
            if targets is None or call.name.upper() in targets:
                call.function = call.function.clone(name=f'{call.name}{self.suffix}')

    def rename_imports(self, source, imports, **kwargs):
        """
        Update imports of actively transformed subroutines.

        :param targets: Optional list of subroutine names for which to
                        modify the corresponding calls.
        """
        targets = as_tuple(kwargs.get('targets', None))
        targets = as_tuple(str(t).upper() for t in targets)

        if self.replace_ignore_items:
            item = kwargs.get('item', None)
            targets += as_tuple(str(i).upper() for i in item.ignore) if item else ()

        # Transformer map to remove any outdated imports
        removal_map = {}

        # We go through the IR, as C-imports can be attributed to the body
        for im in imports:
            if im.c_import:
                target_symbol = im.module.split('.')[0].lower()
                if targets is not None and target_symbol.upper() in targets:
                    if self.mode == 'strict':
                        # Modify the the basename of the C-style header import
                        s = '.'.join(im.module.split('.')[1:])
                        im._update(module=f'{target_symbol}{self.suffix}.{s}')

                    else:
                        # Create a new module import with explicitly qualified symbol
                        new_module = self.derive_module_name(im.module.split('.')[0])
                        new_symbol = Variable(name=f'{target_symbol}{self.suffix}', scope=source)
                        new_import = im.clone(module=new_module, c_import=False, symbols=(new_symbol,))
                        source.spec.prepend(new_import)

                        # Mark current import for removal
                        removal_map[im] = None

            else:
                # Modify module import if it imports any targets
                if targets is not None and any(s in targets for s in im.symbols):
                    # Append suffix to all target symbols
                    symbols = as_tuple(s.clone(name=f'{s.name}{self.suffix}')
                                       if s in targets else s for s in im.symbols)
                    module_name = self.derive_module_name(im.module)
                    im._update(module=module_name, symbols=symbols)

                # TODO: Deal with unqualified blanket imports

        # Apply any scheduled import removals to spec and body
        source.spec = Transformer(removal_map).visit(source.spec)
        if isinstance(source, Subroutine):
            source.body = Transformer(removal_map).visit(source.body)

    def rename_interfaces(self, source, intfs, **kwargs):
        """
        Update explicit interfaces to actively transformed subroutines.
        """

        targets = as_tuple(kwargs.get('targets', None))
        targets = as_tuple(str(t).lower() for t in targets)

        if self.replace_ignore_items and (item := kwargs.get('item', None)):
            targets += as_tuple(str(i).lower() for i in item.ignore)

        # Transformer map to remove any outdated interfaces
        removal_map = {}

        for i in intfs:
            new_imports = ()
            for b in i.body:
                if isinstance(b, Subroutine):
                    if targets is not None and b.name in targets:
                        # Create a new module import with explicitly qualified symbol
                        new_module = self.derive_module_name(b.name)
                        new_symbol = Variable(name=f'{b.name}{self.suffix}', scope=source)
                        new_import = Import(module=new_module, c_import=False, symbols=(new_symbol,))
                        new_imports += as_tuple(new_import)

            if new_imports:
                removal_map[i] = new_imports

        # Apply any scheduled interface removals to spec
        if removal_map:
            source.spec = Transformer(removal_map).visit(source.spec)

    def derive_module_name(self, modname):
        """
        Utility to derive a new module name from `suffix` and `module_suffix`
        """

        # First step through known suffix variants to determine canonical basename
        if modname.lower().endswith(self.suffix.lower()+self.module_suffix.lower()):
            idx = modname.lower().rindex(self.suffix.lower()+self.module_suffix.lower())
        elif modname.lower().endswith(self.suffix.lower()):
            idx = modname.lower().rindex(self.suffix.lower())
        elif modname.lower().endswith(self.module_suffix.lower()):
            idx = modname.lower().rindex(self.module_suffix.lower())
        else:
            idx = len(modname)
        base = modname[:idx]

        # Suffix combination to canonical basename
        if self.module_suffix:
            return f'{base}{self.suffix}{self.module_suffix}'
        return f'{base}{self.suffix}'

    def generate_interfaces(self, source):
        """
        Generate external header file with interface block for this subroutine.
        """
        if isinstance(source, Subroutine):
            # No need to rename here, as this has already happened before
            intfb_path = self.include_path/f'{source.name.lower()}.intfb.h'
            with intfb_path.open('w') as f:
                f.write(fgen(source.interface))

    def module_wrap(self, sourcefile, **kwargs):
        """
        Wrap target subroutines in modules and replace in source file.
        """
        targets = as_tuple(kwargs.get('targets', None))
        targets = as_tuple(str(t).upper() for t in targets)
        item = kwargs.get('item', None)

        module_routines = [r for r in sourcefile.all_subroutines
                           if r not in sourcefile.subroutines]

        for routine in sourcefile.subroutines:
            if routine not in module_routines:
                # Skip member functions
                if item and f'#{routine.name.lower()}' != item.name.lower():
                    continue

                # Skip internal utility routines too
                if routine.name.upper() in targets:
                    continue

                # Create wrapper module and insert into file, replacing the old
                # standalone routine
                modname = f'{routine.name}{self.module_suffix}'
                module = Module(name=modname, contains=Section(body=as_tuple(routine)))
                sourcefile.ir._update(body=as_tuple(
                    module if c is routine else c for c in sourcefile.ir.body
                ))
