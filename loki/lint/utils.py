# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from loki.ir import Comment, CommentBlock, LeafNode
from loki.module import Module
from loki.sourcefile import Sourcefile
from loki.subroutine import Subroutine
from loki.visitors import FindNodes, Transformer


__all__ = ['Fixer', 'get_filename_from_parent', 'is_rule_disabled']


class Fixer:
    """
    Operater class to fix problems reported by fixable rules.
    """

    @classmethod
    def fix_module(cls, module, reports, config):  # pylint: disable=unused-argument
        """
        Call `fix_module` for all rules and apply the transformations.
        """
        # TODO: implement this!
        if reports:
            module._source = None
        return module

    @classmethod
    def fix_subroutine(cls, subroutine, reports, config):
        """
        Call `fix_subroutine` for all rules and apply the transformations.
        """
        mapper = {}
        for report in reports:
            rule_config = config[report.rule.__name__]
            mapper.update(report.rule.fix_subroutine(subroutine, report, rule_config) or {})

        if mapper:
            # Apply the changes and invalidate source objects
            subroutine.spec = Transformer(mapper).visit(subroutine.spec)
            subroutine.body = Transformer(mapper).visit(subroutine.body)
            subroutine._source = None
            parent = subroutine.parent
            while parent is not None:
                parent._source = None
                parent = getattr(parent, 'parent', None)

        return subroutine

    @classmethod
    def fix_sourcefile(cls, sourcefile, reports, config):  # pylint: disable=unused-argument
        """
        Call `fix_sourcefile` for all rules and apply the transformations.
        """
        # TODO: implement this!
        if reports:
            sourcefile._source = None
        return sourcefile

    @classmethod
    def fix(cls, ast, reports, config):
        """
        Attempt to fix problems flagged by fixable rules in the given IR object.

        This routine calls `fix_module`, `fix_subroutine` and `fix_file`
        as applicable for all rules on all entities in the given IR object.

        :param ast: the IR object to be fixed.
        :type ast: :py:class:`Sourcefile`, :py:class:`Module`, or
                   :py:class:`Subroutine`
        :param list reports: the fixable :py:class:`RuleReport` reports.
        :type rule_report: :py:class:`FileReport`
        :param dict config: a `dict` with the config values.

        :return: the modified AST object.
        """

        # Fix on source file level
        if isinstance(ast, Sourcefile):
            # Depth-first traversal
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                for routine in ast.subroutines:
                    cls.fix_subroutine(routine, reports, config)
            if hasattr(ast, 'modules') and ast.modules is not None:
                for module in ast.modules:
                    cls.fix_module(module, reports, config)

            cls.fix_sourcefile(ast, reports, config)

        # Fix on module level
        elif isinstance(ast, Module):
            # Depth-first traversal
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                for routine in ast.subroutines:
                    cls.fix_subroutine(routine, reports, config)

            cls.fix_module(ast, reports, config)

        # Fix on subroutine level
        elif isinstance(ast, Subroutine):
            # Depth-first traversal
            if hasattr(ast, 'members') and ast.members is not None:
                for routine in ast.members:
                    cls.fix_subroutine(routine, reports, config)

            cls.fix_subroutine(ast, reports, config)

        return ast


def get_filename_from_parent(obj):
    """
    Try to determine filename of the source file for an IR object

    It follows ``parent`` attributes until :any:`Sourcefile` is encountered.

    Parameters
    ----------
    obj : :any:`Sourcefile`, :any:`Subroutine` or :any:`Module`
        A source file, module or subroutine object.

    Returns
    -------
    str or NoneType
        The filename if found, else `None`.
    """
    scope = obj
    while hasattr(scope, 'parent') and scope.parent:
        # Go up until we are at Sourcefile level
        scope = scope.parent
    if hasattr(scope, 'path'):
        return scope.path
    return None


_disabled_rules_re = re.compile(r'^\s*!\s*loki-lint\s*:(?:.*?)disable=(?P<rules>[\w\.,]*)')

def is_rule_disabled(ir, identifiers):
    """
    Check if a Linter rule is disabled in the provided context via user annotations

    This looks for comments of the form

    .. code-block:

        ! loki-lint: disable=RuleName

    Where ``RuleName`` is one of the provided :data:`identifiers`.

    If :data:`ir` is a :class:`LeafNode`, only any attached in-line comments
    are checked. If :data:`ir` is any other IR object, the entire subtree below
    this object is searched.

    Parameters
    ----------
    ir : :class:`Node` or :class:`ProgramUnit`
        The IR object for which to check if a rule is disabled
    identifiers : list
        A list of string identifiers via which the rule can be disabled

    Returns
    -------
    bool
        Returns `True` if a rule is disabled, otherwise `False`
    """
    def _match_disabled_rules(comment):
        match = _disabled_rules_re.match(comment.text)
        if match:
            for rule in match.group('rules').split(','):
                if rule in identifiers:
                    return True
        return False

    # If we have a leaf node, we check for in-line comments
    if isinstance(ir, LeafNode):
        if hasattr(ir, 'comment') and ir.comment:
            return _match_disabled_rules(ir.comment)
        return False

    # Otherwise: look in the entire subtree
    for comments in FindNodes((Comment, CommentBlock)).visit(ir):
        for comment in getattr(comments, 'comments', [comments]):
            if _match_disabled_rules(comment):
                return True
    return False
