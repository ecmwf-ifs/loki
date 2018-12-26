import sympy
from sympy.core.cache import cacheit, SYMPY_CACHE_SIZE
from sympy.core.numbers import One as SympyOne, Zero

from loki.tools import as_tuple


__all__ = ['Scalar', 'Array', 'Variable', 'Literal', 'LiteralList',
           'RangeIndex', 'InlineCall', 'indexify', '_symbol_type']


def _symbol_type(cls, name, parent=None):
    """
    Create new type instance from cls and inject symbol name
    """
    # Add the parent-object if it exists (`parent`)
    parent = ('%s%%' % parent) if parent is not None else ''
    name = '%s%s' % (parent, name)
    return type(name, (cls, ), dict(cls.__dict__))


def indexify(expr):
    mapper = {}
    for e in sympy.postorder_traversal(expr):
        try:
            mapper[e] = e.indexify()
        except:
            pass
    return expr.xreplace(mapper)


"""
A global cache of modified symbol class objects
"""
_global_symbol_type = cacheit(_symbol_type)


class Scalar(sympy.Symbol):

    is_Scalar = True
    is_Array = False

    def __new__(cls, *args, **kwargs):
        """
        1st-level variable creation with name injection via the object class
        """
        name = kwargs.pop('name')
        parent = kwargs.pop('parent', None)

        # Name injection for sympy.Symbol (so we can do `a%scalar`)
        if parent is not None:
            name = '%s%%%s' % (parent, name)

        # Create a new object from the static constructor with global caching!
        return Scalar.__xnew_cached_(cls, name, parent=parent)

    def __new_stage2__(cls, name, parent=None):
        """
        2nd-level constructor: arguments to this constructor are used
        for symbolic caching
        """
        # Create a new class object to inject custom variable naming
        # newcls = _symbol_type(cls, name, parent)

        # Setting things here before __init__ forces them
        # to be used for symbolic caching. Thus, `parent` is
        # always used for caching, even if it's not in the name
        newobj = sympy.Symbol.__new__(cls, name)
        newobj.name = name
        newobj.parent = parent
        newobj._type = None

        return newobj

    # Create a globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __init__(self, *args, **kwargs):
        """
        Initialisation of non-cached objects attributes
        """
        self._source = kwargs.pop('source', None)
        self.initial = kwargs.pop('initial', None)
        self._type = kwargs.pop('type', self._type)

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        return self._type


class Array(sympy.Function):

    is_Scalar = False
    is_Array = True

    def __new__(cls, *args, **kwargs):
        """
        1st-level variable creation with name injection via the object class
        """
        if cls == Array:
            # An original constructor invocation
            name = kwargs.pop('name')
            dimensions = kwargs.pop('dimensions', None)
            parent = kwargs.pop('parent', None)

            # Inject the symbol name into the class object.
            # Note, this is the SymPy way to inject custom
            # function naming and ensure symbol caching.
            cls = _global_symbol_type(cls, name, parent)
        else:
            # A reconstruction of an array(function) object,
            # as triggered during symbolic manipulation.
            name = cls.__name__
            dimensions = args
            parent = kwargs.pop('parent', None)

        # Create a new object from the static constructor with global caching!
        return Array.__xnew_cached_(cls, name, dimensions, parent=parent)

    def __new_stage2__(cls, name, dimensions, parent=None):
        """
        2nd-level constructor: arguments to this constructor are used
        for symbolic caching
        """
        # Setting things here before __init__ forces them
        # to be used for symbolic caching. Thus, `parent` is
        # always used for caching, even if it's not in the name
        newobj = sympy.Function.__new__(cls, *dimensions)
        newobj.name = name
        newobj.dimensions = dimensions
        newobj.parent = parent

        newobj._type = None
        newobj._shape = None

        return newobj

    # Use the sympy.core.cache.cacheit decorator to a kernel to create
    # a static globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __init__(self, *args, **kwargs):
        """
        Initialisation of non-cached objects attributes
        """
        self._source = kwargs.pop('source', None)
        self.initial = kwargs.pop('initial', None)
        self._type = kwargs.pop('type', self._type)

    @property
    def type(self):
        """
        Internal representation of the declared data type.
        """
        return self._type

    @property
    def shape(self):
        """
        Original allocated shape of the variable as a tuple of dimensions.
        """
        return self._shape

    def _fcode(self, printer=None):
        """
        Define how we would like to be printed in Fortran code.
        """
        return str(self)

    @property
    def indexed(self):
        name = self.name if self.parent is None else '%s%%%s' % (self.parent, self.name)
        return sympy.IndexedBase(name, shape=self.args)

    def indexify(self):
        return self.indexed[self.args]


class Variable(sympy.Function):
    """
    A symbolic object representing either a :class:`Scalar` or a :class:`Array`
    variable in arithmetic expressions.
    """

    def __new__(cls, *args, **kwargs):
        """
        1st-level variables creation with name injection via the object class
        """
        name = kwargs.pop('name')
        dimensions = kwargs.pop('dimensions', None)
        parent = kwargs.pop('parent', None)

        # Create a new object from the static constructor with global caching!
        if dimensions is None:
            v = Scalar.__new__(Scalar, name=name, parent=parent)
        else:
            v = Array.__new__(Array, name=name, dimensions=dimensions, parent=parent)

        v.__init__(*args, **kwargs)
        return v

    
    @property
    def type(self):
        return self._type

    @property
    def shape(self):
        """
        Original allocated shape of the variable as a tuple of dimensions.
        """
        return self._shape

    @property
    def children(self):
        c = self.dimensions
        if self.parent is not None:
            c += (self.parent, )
        return c


    
class FloatLiteral(sympy.Float):
    __slots__ = ['_mpf_', '_prec','_type', '_kind']

class IntLiteral(sympy.Integer):
    __slots__ = ['p', '_type', '_kind']


class Literal(sympy.Number):

    def __new__(cls, value, **kwargs):
        # We first create a dummy object to determine
        # SymPy's internal literal type, so that we can
        # the create the corrected slotted type for it.

        # Ensure we capture booleans
        if str(value).lower() in ['.true.', '.false.']:
            value = str(value).strip('.').lower()

        # Create a sympyfied dummy to determine type
        dummy = sympy.sympify(value)
        if isinstance(dummy, str):
            return dummy
        elif dummy.is_Integer:
            obj = sympy.Expr.__new__(IntLiteral)
        elif dummy.is_Float:
            obj = sympy.Expr.__new__(FloatLiteral)
        else:
            # We only overload integer and floats
            return dummy

        # Then we copy over the defining slotted attributes
        if isinstance(dummy, SympyOne):
            # One is treated specially in SymPy (as a singletone)
            obj.p = SympyOne.p
        elif isinstance(dummy, Zero):
            # Zero is also treated specially in SymPy
            obj.p = Zero.p
        else:
            for attr in dummy.__class__.__slots__:
                setattr(obj, attr, getattr(dummy, attr))

        # And attach out own meta-data
        obj._type = kwargs.get('type', None)
        obj._kind = kwargs.get('kind', None)
        return obj


class LiteralList(object):

    def __new__(self, values):
        return tuple(Literal(v) for v in values)


class InlineCall(sympy.codegen.ast.FunctionCall):
    """
    Internal representation of an in-line function call
    """
    __slots__ = ['name', 'arguments', 'kwarguments']

    defaults = {'arguments': tuple(), 'kwarguments': dict()}

    def __init__(self, name, arguments=None, kwarguments=None):
        self.name = name
        self.arguments = arguments
        self.kwarguments = kwarguments

    @property
    def function_args(self):
        """
        Construct function arguments (required by sympy code printers)
        """
        kwargs = ()
        if self.kwarguments:
            kwargs = tuple(arg for _, arg in self.kwarguments)
        return as_tuple(self.arguments) + kwargs

    @property
    def args(self):
        """
        Ensure we flatten argumentss and kwarguments for traversal
        """
        return self.function_args

    @property
    def expr(self):
        kwargs = tuple('%s=%s' % (k, v) for k, v in as_tuple(self.kwarguments))
        args = as_tuple(self.arguments) + kwargs
        return '%s(%s)' % (self.name, ','.join(str(a) for a in args))

    def _sympystr(self, printer=None):
        return self.expr

    @property
    def children(self):
        return self.arguments


class RangeIndex(sympy.Idx):

    is_Symbol = True
    is_Function = False

    def __new__(cls, lower=None, upper=None, step=None):
        # Drop trivial default bounds and step sizes
        lower = None if lower == 1 else lower
        step = None if step == 1 else step

        label = ':' if upper is None else str(upper)
        label = label if lower is None else '%s:%s' % (lower, label)
        label = label if step is None else '%s:%s' % (label, step)
        return sympy.Expr.__new__(cls, sympy.Symbol(label))

    def __init__(self, lower=None, upper=None, step=None):
        self._lower = lower
        self._upper = upper
        self._step = step

    @property
    def symbol(self):
        return self.args[0]

    @property
    def name(self):
        return self.symbol.name

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def step(self):
        return self._step
