import sympy
from sympy.core.cache import cacheit, SYMPY_CACHE_SIZE
import fastcache


def _symbol_type(cls, name, parent=None):
    """
    Create new type instance from cls and inject symbol name
    """
    # Add the parent-object if it exists (`parent`)
    parent = ('%s.' % parent) if parent is not None else ''
    name = '%s%s' % (parent, name)
    return type(name, (cls, ), dict(cls.__dict__))


class Var(sympy.Function):

    ### 1st-level variables creation with name injection via the object class
    def __new__(cls, *args, **kwargs):
        name = kwargs.pop('name')
        dims = kwargs.pop('dims')
        parent = kwargs.pop('parent', None)

        # Create a new object from the static constructor with global caching!
        newobj = Var.__xnew_cached_(cls, name, dims, parent=parent)

        return newobj

    def __new_stage2__(cls, name, dims, parent=None):
        # The arguments to this 2nd-level constructor are
        # the ones that are used for symbolic caching
        newcls = _symbol_type(cls, name, parent)

        # Setting things here before __init__ forces them
        # to be used for symbolic caching. Thus, `parent` is
        # always used for caching, even if it's not in the name
        newobj = sympy.Function.__new__(newcls, *dims)
        newobj.name = name
        newobj.dims = dims
        newobj.parent = parent

        print("Created core object: %s" % newobj)
        return newobj

    # Use the sympy.core.cache.cacheit decorator to a kernel to create
    # a static globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __init__(self, *args, **kwargs):
        # Important: Despite the caching in __new__, __init__ will
        # always be executed for each "attempted instantiation",
        # due to the direct inheritance from sympy.Function.__new__. This
        # means repeated assignments here can overwrite previous values.
        self._shape_cache = None
        self.meta = kwargs.get('meta', None)
        print("In __init__, meta %s" % self.meta)

    @property
    def shape(self):
        return self._shape_cache[self.name]

    @shape.setter
    def shape(self, newshape):
        self._shape_cache[self.name] = newshape


###  Now run stuff to demonstrate:

# First, create dummy dimension symbols and plain sympy functions
x, y = sympy.symbols('x y')
f0 = sympy.Function(name='f')(x, y)
f1 = sympy.Function('f')(x, y)

# Demosntrate symbolic equivalence
f0_plus_1 = f0 + 1
print("f0 + 1 == 1 + f1: %s" % (f0 + 1 == 1 + f1))
print("F plus 2: %s" % (f0_plus_1 + 1))


# Add parent symbols and demonstrate how they affect equivalence
a, b = sympy.symbols('a b')
g = Var(name='g', dims=(x, y), parent=a, meta='hi')
g2 = Var(name='g', dims=(x, y), parent=a)
g3 = Var(name='g', dims=(x, y))
print("g: %s => g.parent: %s" % (g, g.parent))
print("g2: %s => g2.parent: %s" % (g2, g2.parent))
print("g3: %s => g3.parent: %s" % (g3, g3.parent))

g_plus_1 = g + 1
print("g plus 2: %s" % (g_plus_1 + 1))

# Symbolic equivalence...
print("g + 1 == 1 + g2: %s" % (g + 1 == 1 + g2))
print("g + 1 == 1 + g3: %s" % (g + 1 == 1 + g3))
print("g2 + 1 == 1 + g3: %s" % (g2 + 1 == 1 + g3))

print("g.meta: %s ... g2.meta: %s" % (g.meta, g2.meta))

g.meta = {'hello': 'world'}
print("g.meta => g2.meta: %s" % g2.meta)


# Now figure out how to scope variables:
# We need to be able to attach info to a variable in scope 1 without
# it affecting variables of the same name and signature in scope 2.

# For this, let's define a kernel class on which to cache....
class Kernel(object):

    def __init__(self):
        # Instantiate our variable cache and wrap the Var constructor in a decorator
        self._varcache = fastcache.clru_cache(SYMPY_CACHE_SIZE, typed=True, unhashable='ignore')(Var.__new_stage2__)
        self._shape_cache = {}

    def Var(self, *args, **kwargs):
        # Here, we emulate Var.__new__, but we call the 2nd stage through
        # the locally cached decorator. This means we need
        name = kwargs.pop('name')
        dims = kwargs.pop('dims')
        parent = kwargs.pop('parent', None)

        # Create a new object from the static constructor with local caching on `Kernel` instance!
        newobj = self._varcache(Var, name, dims, parent=parent)
        # Since we are not actually using the object instation
        # mechanism, we need to call __init__ ourselves.
        newobj.__init__(*args, **kwargs)

        # Now, for a final bit of magic, let's manually track shapes
        #
        # Note that we adopt shape from the first creation
        # of the variable symbol in question. We could force
        # this to happen based on context info, eg. set shape
        # whenever we encounter a variable within a declaration
        # context.
        if not newobj.name in self._shape_cache:
            self._shape_cache[newobj.name] = newobj.dims

        newobj._shape_cache = self._shape_cache

        return newobj


k1 = Kernel()
k2 = Kernel()
kv1 = k1.Var(name='h', dims=(x, y))
kv2 = k2.Var(name='h', dims=(x, y))
print('k1::h:  %s' % kv1)
print('k2::h:  %s' % kv2)

kv1.meta = {'some': 'info'}

print('k1::h %s  =>  k2::h %s' % (kv1.meta, kv2.meta))

# Syncing variable shapes between declaration (first instance)
# and all instances thereafter.
k = Kernel()
v1 = k.Var(name='h', dims=(x, y))
v2 = k.Var(name='h', dims=(x, 1))
print('v1:  %s => shape %s' % (v1, v1.shape))
print('v2:  %s => shape %s' % (v2, v2.shape))

# Now try to change the shape... ?
v1.shape = (y, x)
print('v1:  %s => shape %s' % (v1, v1.shape))
print('v2:  %s => shape %s' % (v2, v2.shape))

from IPython import embed; embed()
