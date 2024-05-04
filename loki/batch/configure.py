# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from fnmatch import fnmatch
from itertools import accumulate
from pathlib import Path
import re

from loki.dimension import Dimension
from loki.tools import as_tuple, CaseInsensitiveDict, load_module
from loki.types import ProcedureType, DerivedType
from loki.logging import error, warning


__all__ = [
    'SchedulerConfig', 'TransformationConfig', 'PipelineConfig',
    'ItemConfig'
]


class SchedulerConfig:
    """
    Configuration object for the :any:`Scheduler`

    It encapsulates config options for scheduler behaviour, with default options
    and item-specific overrides, as well as transformation-specific parameterisations.

    The :any:`SchedulerConfig` can be created either from a raw dictionary or configuration file.

    Parameters
    ----------
    default : dict
        Default options for each item
    routines : dict of dicts or list of dicts
        Dicts with routine-specific options.
    dimensions : dict of dicts or list of dicts
        Dicts with options to define :any:`Dimension` objects.
    disable : list of str
        Subroutine names that are entirely disabled and will not be
        added to either the callgraph that we traverse, nor the
        visualisation. These are intended for utility routines that
        pop up in many routines but can be ignored in terms of program
        control flow, like ``flush`` or ``abort``.
    enable_imports : bool
￼       Disable the inclusion of module imports as scheduler dependencies.
    transformation_configs : dict
        Dicts with transformation-specific options
    frontend_args : dict
        Dicts with file-specific frontend options
    """

    def __init__(
            self, default, routines, disable=None, dimensions=None,
            transformation_configs=None, pipeline_configs=None,
            enable_imports=False, frontend_args=None
    ):
        self.default = default
        self.disable = as_tuple(disable)
        self.dimensions = dimensions
        self.enable_imports = enable_imports

        self.routines = CaseInsensitiveDict(routines)
        self.transformation_configs = transformation_configs
        self.pipeline_configs = pipeline_configs
        self.frontend_args = frontend_args

        # Resolve the dimensions for trafo configurations
        for cfg in self.transformation_configs.values():
            cfg.resolve_dimensions(dimensions)

        # Instantiate Transformation objects
        self.transformations = {
            name: config.instantiate() for name, config in self.transformation_configs.items()
        }

        # Instantiate Pipeline objects
        self.pipelines = {
            name: config.instantiate(transformation_map=self.transformations)
            for name, config in self.pipeline_configs.items()
        }

    @classmethod
    def from_dict(cls, config):
        default = config.get('default', {})
        routines = config.get('routines', [])
        disable = default.get('disable', None)
        enable_imports = default.get('enable_imports', False)

        # Add any dimension definitions contained in the config dict
        dimensions = config.get('dimensions', {})
        dimensions = {k: Dimension(**d) for k, d in dimensions.items()}

        # Create config objects for Transformation configurations
        transformation_configs = config.get('transformations', {})
        transformation_configs = {
            name: TransformationConfig(name=name, **cfg)
            for name, cfg in transformation_configs.items()
        }
        frontend_args = config.get('frontend_args', {})

        pipeline_configs = config.get('pipelines', {})
        pipeline_configs = {
            name: PipelineConfig(name=name, **cfg)
            for name, cfg in pipeline_configs.items()
        }

        return cls(
            default=default, routines=routines, disable=disable, dimensions=dimensions,
            transformation_configs=transformation_configs, pipeline_configs=pipeline_configs,
            frontend_args=frontend_args, enable_imports=enable_imports
        )

    @classmethod
    def from_file(cls, path):
        import toml  # pylint: disable=import-outside-toplevel
        # Load configuration file and process options
        with Path(path).open('r') as f:
            config = toml.load(f)

        return cls.from_dict(config)

    @staticmethod
    def match_item_keys(item_name, keys, use_pattern_matching=False, match_item_parents=False):
        """
        Helper routine to match an item name against config keys.

        The :data:`item_name` may be a fully-qualified name of an :any:`Item`, which may
        include a scope, or only a partial, e.g., local name part. This is then compared
        against a provided list of keys as they may appear in a config property (for
        example an ``ignore`` or ``disable`` list).

        By default, the fully qualified name and the local name are matched.
        Optionally, the matching can be be extended to parent scopes in the item name,
        which is particularly useful if, e.g., the item name of a module member is checked
        against an exclusion list, which lists the module name. This is enabled via
        :data:`match_item_parents`.

        The matching can take patterns in the :data:`keys` into account, allowing for the
        pattern syntax supported by :any:`fnmatch`.
        This requires enabling :data:`use_pattern_matching`.

        Parameters
        ----------
        item_name : str
            The item name to check for matches
        keys : list of str
            The config key values to check for matches
        use_pattern_matching : bool, optional
            Allow patterns in :data:`keys` when matching (default ``False``)
        match_item_parents : bool, optional
            Match also name parts of parent scopes in :data:`item_name`

        Returns
        -------
        tuple of str
            The entries in :data:`keys` that :data:`item_name` matched
        """
        # Sanitize the item name
        item_name = item_name.lower()
        name_parts = item_name.split('#')
        if len(name_parts) == 1:
            scope_name, local_name = '', name_parts[0]
        elif len(name_parts) == 2:
            scope_name, local_name = name_parts
        else:
            raise ValueError(f'Invalid item name {item_name}: More than one `#` in the name.')

        # Build the variations of item name to match
        item_names = {item_name, local_name}
        if match_item_parents:
            if scope_name:
                item_names.add(scope_name)
            if '%' in local_name:
                type_name, *member_names = local_name.split('%')
                item_names |= {
                    name
                    for partial_name in accumulate(member_names, lambda l, r: f'{l}%{r}', initial=type_name)
                    for name in (f'{scope_name}#{partial_name}', partial_name)
                }

        # Match against keys
        keys = as_tuple(keys)
        if use_pattern_matching:
            return tuple(key for key in keys or () if any(fnmatch(name, key.lower()) for name in item_names))
        return tuple(key for key in keys or () if key.lower() in item_names)

    def create_item_config(self, name):
        """
        Create the bespoke config `dict` for an :any:`Item`

        The resulting config object contains the :attr:`default`
        values and any item-specific overwrites and additions.
        """
        keys = self.match_item_keys(name, self.routines)
        if len(keys) > 1:
            if self.default.get('strict'):
                raise RuntimeError(f'{name} matches multiple config entries: {", ".join(keys)}')
            warning(f'{name} matches multiple config entries: {", ".join(keys)}')
        item_conf = self.default.copy()
        for key in keys:
            item_conf.update(self.routines[key])
        return item_conf

    def create_frontend_args(self, path, default_args):
        """
        Create bespoke ``frontend_args`` to pass to the constructor
        or ``make_complete`` method for a file

        The resulting `dict` contains overwrites that have been provided
        in the :attr:`frontend_args` of the config.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path for which to create the frontend arguments. This
            can be a fully-qualified path or include :any:`fnmatch`-compatible
            patterns.
        default_args : dict
            The default options to use. Only keys that are explicitly overriden
            for the file in the scheduler config are updated.

        Returns
        -------
        dict
            The frontend arguments, with file-specific overrides of
            :data:`default_args` if specified in the Scheduler config.
        """
        path = str(path).lower()
        frontend_args = default_args.copy()
        for key, args in (self.frontend_args or {}).items():
            pattern = key.lower() if key[0] == '/' else f'*{key}'.lower()
            if fnmatch(path, pattern):
                frontend_args.update(args)
                return frontend_args
        return frontend_args

    def is_disabled(self, name):
        """
        Check if the item with the given :data:`name` is marked as `disabled`
        """
        return len(self.match_item_keys(name, self.disable, use_pattern_matching=True, match_item_parents=True)) > 0


class TransformationConfig:
    """
    Configuration object for :any:`Transformation` instances that can
    be used to create :any:`Transformation` objects from dictionaries
    or a config file.

    Parameters
    ----------
    name : str
        Name of the transformation object
    module : str
        Python module from which to load the transformation class
    classname : str, optional
        Name of the class to look for when instantiating the transformation.
        If not provided, ``name`` will be used instead.
    path : str or Path, optional
        Path to add to the sys.path before attempting to load the ``module``
    options : dict
        Dicts of options that define the transformation behaviour.
        These options will be passed as constructor arguments using
        keyword-argument notation.
    """

    _re_dimension = re.compile(r'\%dimensions\.(.*?)\%')

    def __init__(self, name, module, classname=None, path=None, options=None):
        self.name = name
        self.module = module
        self.classname = classname or self.name
        self.path = path
        self.options = dict(options)

    def resolve_dimensions(self, dimensions):
        """
        Substitute :any:`Dimension` objects for placeholder strings.

        The format of the string replacement matches the TOML
        configuration.  It will attempt to replace ``%dimensions.dim_name%``
        with a :any:`Dimension` found in :data:`dimensions`:

        Parameters
        ----------
        dimensions : dict
            Dict matching string to pre-configured :any:`Dimension` objects.
        """
        for key, val in self.options.items():
            if not isinstance(val, str):
                continue

            matches = self._re_dimension.findall(val)
            matches = tuple(dimensions[m] for m in as_tuple(matches))
            if matches:
                self.options[key] = matches[0] if len(matches) == 1 else matches

    def instantiate(self):
        """
        Creates instantiated :any:`Transformation` object from stored config options.
        """
        # Load the module that contains the transformations
        mod = load_module(self.module, path=self.path)

        # Check for and return Transformation class
        if not hasattr(mod, self.classname):
            raise RuntimeError(f'Failed to load Transformation class: {self.classname}')

        # Attempt to instantiate transformation from config
        try:
            transformation = getattr(mod, self.classname)(**self.options)
        except TypeError as e:
            error(f'[Loki::Transformation] Failed to instiate {self.classname} from configuration')
            error(f'    Options passed: {self.options}')
            raise e

        return transformation


class PipelineConfig:
    """
    Configuration object for custom :any:`Pipeline` instances that can
    be used to create pipelines from other transformations stored in
    the config.

    Parameters
    ----------
    name : str
        Name of the transformation object
    transformations : list of str
        List of transformation names for which to look when
        instnatiating thie pipeline.
    """


    def __init__(self, name, transformations=None):
        self.name = name
        self.transformations = transformations or []

    def instantiate(self, transformation_map=None):
        """
        Creates a custom :any:`Pipeline` object from instantiated
        :any:`Transformation` or :any:`Pipeline` objects in the given map.
        """
        from loki.batch.pipeline import Pipeline  # pylint: disable=import-outside-toplevel,cyclic-import

        # Create an empty pipeline and add from the map
        pipeline = Pipeline(classes=())
        for name in self.transformations:
            if name not in transformation_map:
                error(f'[Loki::Pipeline] Failed to find {name} in transformation config!')
                raise RuntimeError(f'[Loki::Pipeline] Transformation {name} not found!')

            # Use native notation to append transformation/pipeline,
            # so that we may use them interchangably in config
            pipeline += transformation_map[name]

        return pipeline


class ItemConfig:
    """
    :any:`Item`-specific configuration settings.

    This is filled by inheriting values from :any:`SchedulerConfig.default`
    and applying explicit specialisations provided for an item in the config
    file or dictionary.

    Attributes
    ----------
    role : str or None
        Role in the transformation chain, typically ``'driver'`` or ``'kernel'``
    mode : str or None
        Transformation "mode" to pass to transformations applied to the item
    expand : bool (default: False)
        Flag to enable/disable expansion of children under this node
    strict : bool (default: True)
        Flag controlling whether to fail if dependency items cannot be found
    replicate : bool (default: False)
        Flag indicating whether to mark item as "replicated" in call graphs
    disable : tuple
        List of dependency names that are completely ignored and not reported as
        dependencies by the item. Useful to exclude entire call trees or utility
        routines.
    block : tuple
        List of dependency names that should not be added to the scheduler graph
        as dependencies and are not processed as targets. Note that these might still
        be shown in the graph visualisation.
    ignore : tuple
        List of dependency names that should not be added to the scheduler graph
        as dependencies (and are therefore not processed by transformations)
        but are treated in the current item as targets. This facilitates processing
        across build targets, where, e.g., caller and callee-side are transformed in
        separate Loki passes.
    enrich : tuple
        List of program units that should still be looked up and used to "enrich"
        IR nodes (e.g., :any:`ProcedureSymbol` in :any:`CallStatement`) in this item
        for inter-procedural transformation passes.

    Parameters
    ----------
    config : dict
        The config values for the :any:`Item`. Typically generated by
        :any:`SchedulerConfig.create_item_config`.
    """

    def __init__(self, config):
        self.config = config or {}
        super().__init__()

    @property
    def role(self):
        """
        Role in the transformation chain, for example ``'driver'`` or ``'kernel'``
        """
        return self.config.get('role', None)

    @property
    def mode(self):
        """
        Transformation "mode" to pass to the transformation
        """
        return self.config.get('mode', None)

    @property
    def expand(self):
        """
        Flag to trigger expansion of children under this node
        """
        return self.config.get('expand', False)

    @property
    def strict(self):
        """
        Flag controlling whether to strictly fail if source file cannot be parsed
        """
        return self.config.get('strict', True)

    @property
    def replicate(self):
        """
        Flag indicating whether to mark item as "replicated" in call graphs
        """
        return self.config.get('replicate', False)

    @property
    def disable(self):
        """
        List of sources to completely exclude from expansion and the source tree.
        """
        return self.config.get('disable', tuple())

    @property
    def block(self):
        """
        List of sources to block from processing, but add to the
        source tree for visualisation.
        """
        return self.config.get('block', tuple())

    @property
    def ignore(self):
        """
        List of sources to expand but ignore during processing
        """
        return self.config.get('ignore', tuple())

    @property
    def enrich(self):
        """
        List of sources to to use for IPA enrichment
        """
        return self.config.get('enrich', tuple())

    @property
    def is_ignored(self):
        """
        Flag controlling whether the item is ignored during processing
        """
        return self.config.get('is_ignored', False)

    @classmethod
    def match_symbol_or_name(cls, symbol_or_name, keys, scope=None):
        """
        Match a :any:`TypedSymbol`, :any:`MetaSymbol` or name against
        a list of config values given as :data:`keys`

        This checks whether :data:`symbol_or_name` matches any of the given entries,
        which would typically be something like the :attr:`disable`, :attr:`ignore`, or
        :attr:`block` config entries.

        Optionally, :data:`scope` provides the name of the scope in which
        :data:`symbol_or_name` is defined.
        For derived type members, this takes care of resolving to the type name
        and matching that. This will also match successfully, if only parent components
        match, e.g., the scope name or the type name of the symbol.
        The use of simple patterns is allowed, see :any:`SchedulerConfig.match_item_keys`
        for more information.

        Parameters
        ----------
        symbol_or_name : :any:`TypedSymbol` or :any:`MetaSymbol` or str
            The symbol or name to match
        keys : list of str
            The list of candidate names to match against. This can be fully qualified
            names (e.g., ``'my_scope#my_routine'``), plain scope or routine names
            (e.g., ``'my_scope'`` or ``'my_routine'``), or use simple patterns (e.g., ``'my_*'``).
        scope : str, optional
            The name of the scope, in which :data:`symbol_or_name` is defined, if available.
            Providing this allows to match a larger range of name combinations

        Returns
        -------
        bool
            ``True`` if matched successfully, otherwise ``False``
        """
        if isinstance(symbol_or_name, str):
            scope_prefix = f'{scope!s}#'.lower() if scope is not None else ''
            return len(SchedulerConfig.match_item_keys(
                f'{scope_prefix}{symbol_or_name}', keys, use_pattern_matching=True, match_item_parents=True
            )) > 0

        if parents := getattr(symbol_or_name, 'parents', None):
            type_name = parents[0].type.dtype.name
            parents = [parent.basename for parent in parents[1:]]
            return cls.match_symbol_or_name(
                '%'.join([type_name, *parents, symbol_or_name.basename]), keys, scope=scope
            )

        if type_ := getattr(symbol_or_name, 'type', None):
            if isinstance(type_.dtype, (ProcedureType, DerivedType)):
                return cls.match_symbol_or_name(type_.dtype.name, keys, scope=scope)

        return cls.match_symbol_or_name(str(symbol_or_name), keys, scope=scope)
