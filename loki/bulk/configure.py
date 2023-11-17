# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from collections import OrderedDict

from loki.dimension import Dimension
from loki.tools import as_tuple, CaseInsensitiveDict


__all__ = ['SchedulerConfig']


class SchedulerConfig:
    """
    Configuration object for the transformation :any:`Scheduler` that
    encapsulates default behaviour and item-specific behaviour. Can
    be create either from a raw dictionary or configration file.

    Parameters
    ----------
    default : dict
        Default options for each item
    routines : dict of dicts or list of dicts
        Dicts with routine-specific options.
    dimensions : dict of dicts or list of dicts
        Dicts with options to define :any`Dimension` objects.
    disable : list of str
        Subroutine names that are entirely disabled and will not be
        added to either the callgraph that we traverse, nor the
        visualisation. These are intended for utility routines that
        pop up in many routines but can be ignored in terms of program
        control flow, like ``flush`` or ``abort``.
    enable_imports : bool
        Disable the inclusion of module imports as scheduler dependencies.
    """

    def __init__(self, default, routines, disable=None, dimensions=None, dic2p=None, derived_types=None,
                 enable_imports=False):
        self.default = default
        if isinstance(routines, dict):
            self.routines = CaseInsensitiveDict(routines)
        else:
            self.routines = CaseInsensitiveDict((r.name, r) for r in as_tuple(routines))
        self.disable = as_tuple(disable)
        self.dimensions = dimensions
        self.enable_imports = enable_imports

        if dic2p is not None:
            self.dic2p = dic2p
        else:
            self.dic2p = {}
        if derived_types is not None:
            self.derived_types = derived_types
        else:
            self.derived_types = ()

    @classmethod
    def from_dict(cls, config):
        default = config['default']
        if 'routine' in config:
            config['routines'] = OrderedDict((r['name'], r) for r in config.get('routine', []))
        else:
            config['routines'] = []
        routines = config['routines']
        disable = default.get('disable', None)
        enable_imports = default.get('enable_imports', False)

        # Add any dimension definitions contained in the config dict
        dimensions = {}
        if 'dimension' in config:
            dimensions = [Dimension(**d) for d in config['dimension']]
            dimensions = {d.name: d for d in dimensions}

        dic2p = {}
        if 'dic2p' in config:
            dic2p = config['dic2p']

        derived_types = ()
        if 'derived_types' in config:
            derived_types = config['derived_types']

        return cls(default=default, routines=routines, disable=disable, dimensions=dimensions, dic2p=dic2p,
                   derived_types=derived_types, enable_imports=enable_imports)

    @classmethod
    def from_file(cls, path):
        import toml  # pylint: disable=import-outside-toplevel
        # Load configuration file and process options
        with Path(path).open('r') as f:
            config = toml.load(f)

        return cls.from_dict(config)
