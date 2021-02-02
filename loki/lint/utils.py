from loki import Sourcefile, Module, Subroutine, Transformer

__all__ = ['Fixer', 'get_filename_from_parent']


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
                ast._routines = [cls.fix(routine, reports, config) for routine in ast.subroutines]
            if hasattr(ast, 'modules') and ast.modules is not None:
                ast._modules = [cls.fix(module, reports, config) for module in ast.modules]

            ast = cls.fix_sourcefile(ast, reports, config)

        # Fix on module level
        elif isinstance(ast, Module):
            # Depth-first traversal
            if hasattr(ast, 'subroutines') and ast.subroutines is not None:
                ast.routines = [cls.fix(routine, reports, config) for routine in ast.subroutines]

            ast = ast.fix_module(ast, reports, config)

        # Fix on subroutine level
        elif isinstance(ast, Subroutine):
            # Depth-first traversal
            if hasattr(ast, 'members') and ast.members is not None:
                ast._members = [cls.fix(member, reports, config) for member in ast.members]

            ast = cls.fix_subroutine(ast, reports, config)

        return ast


def get_filename_from_parent(obj):
    """
    Try to determine filename by following ``parent`` attributes
    until :py:class:``loki.sourcefile.Sourcefile`` is encountered.

    :param obj: A source file, module or subroutine object.
    :return: The filename or ``None``
    :rtype: str or NoneType
    """
    scope = obj
    while hasattr(scope, 'parent') and scope.parent:
        # Go up until we are at Sourcefile level
        scope = scope.parent
    if hasattr(scope, 'path'):
        return scope.path
    return None
