import os
from collections import OrderedDict


class Configuration(OrderedDict):
    """
    Dictionary class that holds global configuration parameters.

    In addition ato sanity checking this dict also allows callbacks
    to be used to propagate values to the relevant parts of the
    system.

    Example usage:
    ```
    config = Configuration('Loki')
    config.register('log-level', 'INFO', env_variable='LOKI_LOGGING')
    ...

    config.initialize()
    logging = config['log-level']
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

        self._defaults = {}
        self._env_variables = {}
        self._preprocess_functions = {}
        self._callback_functions = {}

    def initialize(self):
        """
        Initialize all registered entries by either using the value given
        via environemnt variables or the default.
        """
        for key in self.keys():
            if self._env_variables[key]:
                env_val = os.environ.get(self._env_variables[key], None)
                if env_val is None:
                    self[key] = self._defaults[key]
                else:
                    self[key] = env_val

    def register(self, key, default, env_variable=None, preprocess=None, callback=None):
        """
        Register configuration option with optional default value
        and callback function.

        Parameters:
        ==========
        * ``key``: Internal name of the configuration option
        * ``default`: Default value if unspecified in environment
        * ``env_variable``: Name of environment variable to check for value
        * ``preprocess``: Optional preprocess function that turns string-based
                          values into the correct format (eg. for env variables)
        * ``callback``: Optional callback function to trigger on updates
        """
        super().__setitem__(key, default)

        self._defaults[key] = default
        self._env_variables[key] = env_variable
        self._preprocess_functions[key] = preprocess
        self._callback_functions[key] = callback

    def print_state(self):
        """
        Print the current configuration state.
        """
        from loki.logging import info  # pylint: disable=import-outside-toplevel
        info("[Loki] global config:")
        for k, v in self.items():
            info('  %s: %s' % (k, v))

    def _updated(self, key, value):
        # Execute callback function for ``key``
        if self._callback_functions[key]:
            self._callback_functions[key](value)

    def __setitem__(self, key, value):
        # Preprocess any given value
        if self._preprocess_functions[key]:
            value = self._preprocess_functions[key](value)

        super().__setitem__(key, value)

        # Trigger configured callbacks
        self._updated(key, value)


config = Configuration('Loki configuration')
