from collections import Iterable


__all__ = ['as_tuple', 'filter_ordered', 'flatten']


def as_tuple(item, type=None, length=None):
    """
    Force item to a tuple.

    Partly extracted from: https://github.com/OP2/PyOP2/.
    """
    # Empty list if we get passed None
    if item is None:
        t = ()
    elif isinstance(item, str):
        t = (item,)
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except (TypeError, NotImplementedError):
            t = (item,) * (length or 1)
    if length and not len(t) == length:
        raise ValueError("Tuple needs to be of length %d" % length)
    if type and not all(isinstance(i, type) for i in t):
        raise TypeError("Items need to be of type %s" % type)
    return t


def filter_ordered(elements, key=None):
    """
    Filter elements in a list while preserving order.

    Partly extracted from: https://github.com/opesci/devito.

    :param key: Optional conversion key used during equality comparison.
    """
    seen = set()
    if key is None:
        key = lambda x: x
    return [e for e in elements if not (key(e) in seen or seen.add(key(e)))]


def flatten(l):
    """
    Flatten a hierarchy of nested lists into a plain list.

    Partly extracted from: https://github.com/opesci/devito.
    """
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist
