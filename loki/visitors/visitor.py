"""
Visitor base classes for traversing the IR
"""

import inspect

__all__ = ['GenericVisitor', 'Visitor']


class GenericVisitor:
    """
    A generic visitor class to traverse the IR tree.

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.
    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.
    The method signature is:

    .. code-block::

       def visit_Foo(self, o, [*args, **kwargs]):
           pass

    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:

    .. code-block::

        def visit_Foo(self, o, parent=None, *args, **kwargs):
            pass
    """

    def __init__(self):
        handlers = {}
        # visit methods are spelt visit_Foo.
        prefix = "visit_"
        # Inspect the methods on this instance to find out which
        # handlers are defined.
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            # Check the argument specification
            # Valid options are:
            #    visit_Foo(self, o, [*args, **kwargs])
            argspec = inspect.getfullargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    default_args = {}
    """
    Dict of default keyword arguments for the visitor. These are not used by
    default in :meth:`visit`, however, a caller may pass them explicitly to
    :meth:`visit` by accessing :attr:`default_args`. For example:

    .. code-block:: python

       v = FooVisitor()
       v.visit(node, **v.default_args)
    """

    @classmethod
    def default_retval(cls):
        """
        Default return value for handler methods.

        This method returns an object to use to populate return values.
        If your visitor combines values in a tree-walk, it may be useful to
        provide an object to combine the results into. :meth:`default_retval`
        may be defined by the visitor to be called to provide an empty object
        of appropriate type.

        Returns
        -------
        None
        """
        return None

    def lookup_method(self, instance):
        """Look up a handler method for a visitee.

        :param instance: The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            # Do we have a method handler defined for this type name
            return self._handlers[cls.__name__]
        except KeyError:
            # No, walk the MRO.
            for klass in cls.mro()[1:]:
                entry = self._handlers.get(klass.__name__)
                if entry:
                    # Save it on this type name for faster lookup next time
                    self._handlers[cls.__name__] = entry
                    return entry
        raise RuntimeError(f'No handler found for class {cls.__name__}')

    def visit(self, o, *args, **kwargs):
        """
        Apply this :class:`Visitor` to an IR tree.

        Parameters
        ----------
        o : :any:`Node`
            The node to visit.
        *args :
            Optional arguments to pass to the visit methods.
        **kwargs :
            Optional keyword arguments to pass to the visit methods.
        """
        meth = self.lookup_method(o)
        return meth(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):  # pylint: disable=unused-argument
        """
        Fallback method for objects that do not match any handler.

        Parameters
        ----------
        o :
            The object to visit.
        **kwargs :
            Optional keyword arguments passed to the visit methods.

        Returns
        -------
        :py:meth:`GenericVisitor.default_retval`
            The default return value.
        """
        return self.default_retval()


class Visitor(GenericVisitor):
    """
    The basic visitor-class for traversing Loki's control flow tree.

    It enhances the generic visitor class :class:`GenericVisitor` with the
    ability to recurse for all children of a :any:`Node`.
    """

    def visit_tuple(self, o, **kwargs):
        """
        Visit all elements in a tuple and return the results as a tuple.
        """
        return tuple(self.visit(c, **kwargs) for c in o)

    visit_list = visit_tuple

    def visit_Node(self, o, **kwargs):
        """
        Visit all children of a :any:`Node`.
        """
        return self.visit(o.children, **kwargs)

    @staticmethod
    def reuse(o, *args, **kwargs):  # pylint: disable=unused-argument
        """A visit method to reuse a node, ignoring children."""
        return o

    def maybe_rebuild(self, o, *args, **kwargs):
        """A visit method that rebuilds nodes if their children have changed."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        if all(a is b for a, b in zip(ops, new_ops)):
            return o
        return o._rebuild(*new_ops, **okwargs)

    def always_rebuild(self, o, *args, **kwargs):
        """A visit method that always rebuilds nodes."""
        ops, okwargs = o.operands()
        new_ops = [self.visit(op, *args, **kwargs) for op in ops]
        return o._rebuild(*new_ops, **okwargs)
