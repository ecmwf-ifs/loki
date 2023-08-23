# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Base class for linter rules and available rule types
"""
from enum import Enum

from loki.lint.utils import is_rule_disabled
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine


class RuleType(Enum):
    """
    Available types for rules with increasing severity.
    """

    INFO = 1
    WARN = 2
    SERIOUS = 3
    ERROR = 4


class GenericRule:
    """
    Generic interface for linter rules providing default values and the
    general :meth:`check` routine that calls the specific entry points to rules
    (subroutines, modules, and the source file).

    When adding a new rule, it must inherit from :any:`GenericRule`
    and define :data:`type` and provide ``title`` (and ``id``, if applicable)
    in :data:`docs`.
    Optional configuration values can be defined in :data:`config` together with
    the default value for this option. Only the relevant entry points to a
    rule must be implemented.
    """

    type = None
    """
    The rule type as one of the categories in :any:`RuleType`
    """

    docs = None
    """
    :any:`dict` with description of the rule

    Typically, this should include ``"id"`` and ``"title"``. Allows for
    Python's format specification mini-language in ``"title"`` to fill values
    using data from :data:`config`, with the field name corresponding to the
    config key.
    """

    config = {}
    """
    Dict of configuration keys and their default values

    These values can be overriden externally in the linter config file and are
    passed automatically to the :meth:`check` routine.
    """

    fixable = False
    """
    Indicator for a fixable rule that implements a corresponding :meth:`fix`
    routine
    """

    deprecated = False
    """
    Indicator for a deprecated rule
    """

    replaced_by = ()
    """
    List of rules that replace the deprecated rule, where applicable
    """

    @classmethod
    def identifiers(cls):
        """
        Return list of strings that identify this rule
        """
        if cls.docs and 'id' in cls.docs:  # pylint: disable=unsupported-membership-test
            return [cls.__name__, cls.docs['id']]  # pylint: disable=unsubscriptable-object
        return [cls.__name__]

    @classmethod
    def check_module(cls, module, rule_report, config):
        """
        Perform rule checks on module level

        Must be implemented by a rule if applicable.
        """

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        Perform rule checks on subroutine level

        Must be implemented by a rule if applicable.
        """

    @classmethod
    def check_file(cls, sourcefile, rule_report, config):
        """
        Perform rule checks on file level

        Must be implemented by a rule if applicable.
        """

    @classmethod
    def check(cls, ast, rule_report, config, **kwargs):
        """
        Perform checks on all entities in the given IR object

        This routine calls :meth:`check_module`, :meth:`check_subroutine`
        and :meth:`check_file` as applicable for all entities in the given
        IR object.

        Parameters
        ----------
        ast : :any:`Sourcefile` or :any:`Module` or :any:`Subroutine`
            The IR object to be checked.
        rule_report : :any:`RuleReport`
            The reporter object in which rule violations should be registered.
        config : dict
            The rule configuration, filled with externally provided
            configuration values or the rule's default configuration.
        """

        # Perform checks on source file level
        if isinstance(ast, Sourcefile):
            cls.check_file(ast, rule_report, config)

            # Then recurse for all modules and subroutines in that file
            if hasattr(ast, 'modules') and ast.modules is not None:
                for module in ast.modules:
                    cls.check(module, rule_report, config, **kwargs)
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                for subroutine in ast.subroutines:
                    cls.check(subroutine, rule_report, config, **kwargs)

        # Perform checks on module level
        elif isinstance(ast, Module):
            if is_rule_disabled(ast.spec, cls.identifiers()):
                return

            cls.check_module(ast, rule_report, config)

            # Then recurse for all subroutines in that module
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                for subroutine in ast.subroutines:
                    cls.check(subroutine, rule_report, config, **kwargs)

        # Peform checks on subroutine level
        elif isinstance(ast, Subroutine):
            if is_rule_disabled(ast.ir, cls.identifiers()):
                return

            if not (targets := kwargs.pop('targets', None)):
                items = kwargs.get('items', ())
                item = [item for item in items if item.local_name.lower() == ast.name.lower()]
                if len(item) > 0:
                    targets = item[0].targets
            cls.check_subroutine(ast, rule_report, config, targets=targets, **kwargs)

            # Recurse for any procedures contained in a subroutine
            if hasattr(ast, 'members') and ast.members is not None:
                for member in ast.members:
                    cls.check(member, rule_report, config, **kwargs)

    @classmethod
    def fix_module(cls, module, rule_report, config):
        """
        Fix rule violations on module level

        Must be implemented by a rule if applicable.
        """

    @classmethod
    def fix_subroutine(cls, subroutine, rule_report, config):
        """
        Fix rule violations on subroutine level

        Must be implemented by a rule if applicable.
        """

    @classmethod
    def fix_sourcefile(cls, sourcefile, rule_report, config):
        """
        Fix rule violations on sourcefile level

        Must be implemented by a rule if applicable.
        """
