from pathlib import Path

from loki.transform.transformation import Transformation
from loki.subroutine import Subroutine
from loki.module import Module
from loki.visitors import FindNodes, Transformer
from loki.ir import CallStatement, Import
from loki.backend import fgen
from loki.tools import as_tuple


__all__ = ['DependencyTransformation']


class DependencyTransformation(Transformation):
    """
    Basic `Transformation` class that facilitates dependency injection
    for transformed `Modules` and `Subroutines` into complex source
    trees. It does so by appending a provided `suffix` argument to
    transformed subroutine and module objects and changing the target
    names of `Import` and `CallStatement` nodes on the call-site
    accordingly.

    :param suffix: The suffix to apply during renaming
    :param mode: The injection mode to use; either `'strict'` or `'module'`
    :param module_suffix: Special suffix to signal module names like `_MOD`
    :param include path: Directory for generating additional header files

    The `DependencyTransformation` provides two `mode` options:
      * `strict` honors dependencies via C-style headers
      * `module` replaces C-style header dependencies with explicit module imports

    When applying the transformation to a source object, one of two
    "roles" can be specified via the `role` keyword:
      * `driver`: Only renames imports and calls to kernel routines
      * `kernel`: Renames routine or enclosing modules, as well as
                  renaming any further imports and calls.

    Note that `routine.apply(transformation, role='driver')` entails that the
    `routine` still mimicks its original counterpart and can therefore be used
    as a drop-in replacement during compilation that then diverts the
    dependency tree to the modified sub-tree.
    """

    def __init__(self, suffix, mode='module', module_suffix=None, include_path=None):
        self.suffix = suffix
        assert mode in ['strict', 'module']
        self.mode = mode

        self.module_suffix = module_suffix
        self.include_path = None if include_path is None else Path(include_path)

    def transform_subroutine(self, routine, **kwargs):
        """
        Rename driver subroutine and all calls to target routines. In
        'strict' mode, also  re-generate the kernel interface headers.
        """
        role = kwargs.get('role')

        if role == 'kernel':
            # Change the name of kernel routines
            routine.name += self.suffix

        self.rename_calls(routine, **kwargs)

        # Note, C-style imports can be in the body, so use whole IR
        imports = FindNodes(Import).visit(routine.ir)
        self.rename_imports(routine, imports=imports, **kwargs)

        if role == 'kernel' and self.mode == 'strict':
            # Re-generate C-style interface header
            self.generate_interfaces(routine, **kwargs)

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
        targets = kwargs.get('targets', None)
        for call in FindNodes(CallStatement).visit(routine.body):
            if targets is None or call.name in targets:
                call._update(name='{}{}'.format(call.name, self.suffix))

    def rename_imports(self, source, imports, **kwargs):
        """
        Update imports of actively transformed subroutines.

        :param targets: Optional list of subroutine names for which to
                        modify the corresponding calls.
        """
        targets = kwargs.get('targets', None)
        if targets is not None:
            targets = as_tuple(str(t).lower() for t in as_tuple(targets))

        # Transformer map to remove any outdated imports
        removal_map = {}

        # We go through the IR, as C-imports can be attributed to the body
        for im in imports:
            if im.c_import:
                target_symbol = im.module.split('.')[0].lower()
                if targets is not None and target_symbol in targets:
                    if self.mode == 'strict':
                        # Modify the the basename of the C-style header import
                        im._update(module='{}{}.{}'.format(target_symbol, self.suffix,
                                                          '.'.join(im.module.split('.')[1:])))

                    else:
                        # Create a new module import with explicitly qualified symbol
                        new_module = self.derive_module_name(im.module.split('.')[0].upper())
                        new_symbol = '{}{}'.format(target_symbol, self.suffix)
                        new_import = im.clone(module=new_module, c_import=False, symbols=(new_symbol,))
                        source.spec.prepend(new_import)

                        # Mark current import for removal
                        removal_map[im] = None

            else:
                # Modify module import if it imports any targets
                if targets is not None and any(s.lower() in targets for s in im.symbols):
                    # Append suffix to all target symbols
                    symbols = as_tuple('{}{}'.format(s, self.suffix) if s.lower() in targets else s
                                       for s in im.symbols)
                    module_name = self.derive_module_name(im.module)
                    im._update(module=module_name, symbols=symbols)

                # TODO: Deal with unqualified blanket imports

        # Apply any scheduled import removals to spec and body
        source.spec = Transformer(removal_map).visit(source.spec)
        if isinstance(source, Subroutine):
            source.body = Transformer(removal_map).visit(source.body)

    def derive_module_name(self, modname):
        """
        Utility to derive a new module name from `suffix` and `module_suffix`
        """
        if self.module_suffix:
            # If a module suffix is provided, we insert suffix before that
            if self.module_suffix in modname:
                idx = modname.index(self.module_suffix)
                return '{}{}{}'.format(modname[:idx], self.suffix, self.module_suffix)

            return '{}{}{}'.format(modname, self.suffix, self.module_suffix)
        else:
            return '{}{}'.format(modname, self.suffix)

    def generate_interfaces(self, source, **kwargs):
        """
        Generate external header file with interface block for this subroutine.
        """
        if isinstance(source, Subroutine):
            # No need to rename here, as this has already happened before
            intfb_path = self.include_path/'{}.intfb.h'.format(source.name.lower())
            with intfb_path.open('w') as f:
                f.write(fgen(source.interface))

    def module_wrap(self, sourcefile, **kwargs):
        """
        Wrap target subroutines in modules and replace in source file.
        """
        targets = kwargs.get('targets', None)

        module_routines = [r for r in sourcefile.all_subroutines
                           if r not in sourcefile.subroutines]

        for routine in sourcefile.subroutines:
            if routine not in module_routines:
                if targets is None or routine.name+self.suffix in targets:
                    # Create wrapper module and insert into file
                    modname = '{}{}'.format(routine.name, self.module_suffix)
                    module = Module(name=modname, routines=[routine])
                    sourcefile._modules += (module, )

                    # Remove old standalone routine
                    sourcefile._routines = as_tuple(r for r in sourcefile.subroutines
                                                    if r is not routine)
