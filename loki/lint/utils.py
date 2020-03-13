def get_filename_from_parent(obj):
    '''Try to determine filename by following ``parent`` attributes
    until :py:class:``loki.sourcefile.SourceFile`` is encountered.

    :param obj: A source file, module or subroutine object.
    :return: The filename or ``None``
    :rtype: str or NoneType
    '''
    scope = obj
    while hasattr(scope, 'parent') and scope.parent:
        # Go up until we are at SourceFile level
        scope = scope.parent
    if hasattr(scope, 'path'):
        return scope.path
    return None
