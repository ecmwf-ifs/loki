import sympy
from sympy.core.cache import cacheit, SYMPY_CACHE_SIZE
from sympy.logic.boolalg import Boolean, as_Boolean
from sympy.codegen.ast import String
from sympy.printing.codeprinter import CodePrinter
from fastcache import clru_cache

from loki.tools import as_tuple
from loki.types import DataType


__all__ = ['Scalar', 'Array', 'Variable', 'Literal', 'LiteralList',
           'RangeIndex', 'InlineCall', 'Cast', 'indexify', 'SymbolCache']


def _symbol_type(cls, name, parent=None):
    """
    Create new type instance from cls and inject symbol name
    """
    # Add the parent-object if it exists (`parent`)
    parent = ('%s%%' % parent) if parent is not None else ''
    name = '%s%s' % (parent, name)
    return type(name, (cls, ), dict(cls.__dict__))


def indexify(expr, evaluate=True):
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



class SymbolCache(object):
    """
    A specialised cache object for use by subroutine/kernel containers
    that provides cached symbols (scalar and array variables), while
    providing control over the scope of symbolc caching (eg. per
    routine, rather than global). It basically wraps the low-level
    caches for each symbol contructor.
    """

    def __init__(self):
        # Instantiate local symbol caches
        self._symbol_type_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                             unhashable='ignore')(_symbol_type)
        self._array_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                       unhashable='ignore')(Array.__new_stage2__)
        self._scalar_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                        unhashable='ignore')(Scalar.__new_stage2__)

    def Variable(self, *args, **kwargs):
        """
        Variable constructor (phase 1) that uses a specific set of local caches,
        so that symbol caching is scoped to the :class:`SymbolCache` instance.

        Note, that this is the equivalent to the phase 1 constructors `Scalar.__new__`
        and `Array.__new__`, only in a locally caching context.
        """
        name = kwargs.pop('name')
        dimensions = kwargs.pop('dimensions', None)
        parent = kwargs.pop('parent', None)

        # Create scalar or array symbol using local caches
        if dimensions is None:
            cls = self._symbol_type_cache(Scalar, name, parent)
            newobj = self._scalar_cache(cls, name, parent=parent)
        else:
            _type = kwargs.get('type', None)
            if _type and _type.dtype == DataType.BOOL:
                cls = self._symbol_type_cache(BoolArray, name, parent)
            else:
                cls = self._symbol_type_cache(Array, name, parent)
            newobj = self._array_cache(cls, name, dimensions, parent=parent)

        # Since we are not actually using the object instation
        # mechanism, we need to call __init__ ourselves.
        newobj.__init__(*args, **kwargs)
        return newobj


class Scalar(sympy.Symbol):

    is_Scalar = True
    is_Array = False

    def __new__(cls, *args, **kwargs):
        """
        1st-level variable creation with name injection via the object class
        """
        name = kwargs.pop('name', args[0] if len(args) > 0 else None)
        parent = kwargs.pop('parent', None)
        # Name injection for symbols via cls (so we can do `a%scalar`)
        cls = _global_symbol_type(Scalar, name, parent)

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
        # Use cls.__name__ to get the injected name modifcation
        newobj.name = cls.__name__
        newobj.basename = name
        newobj.parent = parent

        newobj._type = None
        newobj.initial = None
        newobj._source = None

        return newobj

    # Create a globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __init__(self, *args, **kwargs):
        """
        Initialisation of non-cached objects attributes
        """
        self._source = kwargs.pop('source', None) or self._source
        self.initial = kwargs.pop('initial', None) or self.initial
        self._type = kwargs.pop('type', None) or self._type

    def clone(self, **kwargs):
        """
        Replicate the :class:`Array` variable with the provided overrides.

        Note, if :param dimensions: is provided as ``None``, as
        :class:`Scalar` variable will be created.

        Alos, it needs a :param cache: to force the correct caching scope.
        """
        # Add existing meta-info to the clone arguments, only if have
        # them. Note, we need to be very careful not to pass eg,
        # `type=None` down, as it might invalidate cached info.
        if self.name and 'name' not in kwargs:
            kwargs['name'] = self.name
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent
        if self.type and 'type' not in kwargs:
            kwargs['type'] = self.type
        if self.initial and 'initial' not in kwargs:
            kwargs['initial'] = self.initial

        cache = kwargs.pop('cache', None)
        if cache is None:
            return Variable(**kwargs)
        else:
            return cache.Variable(**kwargs)

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

            _type = kwargs.get('type', None)
            if _type and _type.dtype == DataType.BOOL:
                cls = _global_symbol_type(BoolArray, name, parent)
            else:
                cls = _global_symbol_type(Array, name, parent)
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
        newobj.name = cls.__name__
        newobj.basename = name
        newobj.dimensions = dimensions
        newobj.parent = parent

        newobj._type = None
        newobj._shape = None
        newobj.initial = None
        newobj._source = None

        return newobj

    # Use the sympy.core.cache.cacheit decorator to a kernel to create
    # a static globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __init__(self, *args, **kwargs):
        """
        Initialisation of non-cached objects attributes
        """
        self._source = kwargs.pop('source', None) or self._source
        self.initial = kwargs.pop('initial', None) or self.initial
        self._type = kwargs.pop('type', None) or self._type
        self._shape = kwargs.pop('shape', None) or self._shape

    def clone(self, **kwargs):
        """
        Replicate the :class:`Array` variable with the provided overrides.

        Note, if :param dimensions: is provided as ``None``, as
        :class:`Scalar` variable will be created.

        Alos, it needs a :param cache: to force the correct caching scope.
        """
        # Add existing meta-info to the clone arguments, only if have
        # them. Note, we need to be very careful not to pass eg,
        # `type=None` down, as it might invalidate cached info.
        if self.name and 'name' not in kwargs:
            kwargs['name'] = self.basename
        if self.dimensions and 'dimensions' not in kwargs:
            kwargs['dimensions'] = self.dimensions
        if self.parent and 'parent' not in kwargs:
            kwargs['parent'] = self.parent
        if self.type and 'type' not in kwargs:
            kwargs['type'] = self.type
        if self.shape and 'shape' not in kwargs:
            kwargs['shape'] = self.shape
        if self.initial and 'initial' not in kwargs:
            kwargs['initial'] = self.initial

        cache = kwargs.pop('cache', None)
        if cache is None:
            return Variable(**kwargs)
        else:
            return cache.Variable(**kwargs)

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
        return sympy.IndexedBase(self.name, shape=self.shape or self.args)

    def indexify(self):
        """
        Return an instance of :class:`sympy.Indexed` that corresponds
        to this :class:`Array` variable. This is useful for generating
        code for array accesses using the sympy code printers.
        """
        return self.indexed[self.args]


class BoolArray(Array, Boolean):
    """
    A specialised form of an :class:`Array` variable that behaves like
    a :class:`sympy.logic.boolargl.Boolean`, allowing us to use it in
    logical expressions.
    """

    def indexify(self):
        """
        Return an instance of :class:`sympy.Indexed` that corresponds
        to this :class:`Array` variable. This is useful for generating
        code for array accesses using the sympy code printers.
        """
        return BoolIndexed(self.indexed, *self.args)

    @property
    def binary_symbols(self):
        # Suppress handing any of our args back to the logic engine,
        # as they are not booleans.
        return set()


class BoolIndexed(sympy.Indexed, Boolean):
    """
    A specialised form of a :class:`sympy.Indexed` symbol that behaves
    like a :class:`sympy.logic.boolargl.Boolean`, allowing us to use
    it in logical expressions.
    """

    @property
    def binary_symbols(self):
        # Suppress handing any of our args back to the logic engine,
        # as they are not booleans.
        return set()


class Variable(sympy.Function):
    """
    A symbolic object representing either a :class:`Scalar` or a :class:`Array`
    variable in arithmetic expressions.

    Note, that this is only a convenience constructor that always returns either
    a :class:`Scalar` or :class:`Array` variable object.
    """

    def __new__(cls, *args, **kwargs):
        """
        1st-level variables creation with name injection via the object class
        """
        name = kwargs.pop('name')
        dimensions = kwargs.pop('dimensions', None)
        parent = kwargs.pop('parent', None)
        _type = kwargs.get('type', None)

        # Create a new object from the static constructor with global caching!
        if dimensions is None:
            v = Scalar.__new__(Scalar, name=name, parent=parent)
        else:
            v = Array.__new__(Array, name=name, dimensions=dimensions, parent=parent, type=_type)

        v.__init__(*args, **kwargs)
        return v

    
class FloatLiteral(sympy.Float):
    __slots__ = ['_mpf_', '_prec','_type', '_kind']

    def _fcode(self, printer=None):
        printed = CodePrinter._print_Float(printer, self)
        if hasattr(self, '_kind') and self._kind is not None:
            return '%s_%s' % (printed, self._kind)
        else:
            return printed


class IntLiteral(sympy.Integer):
    __slots__ = ['p', '_type', '_kind']


class Literal(sympy.Number):

    def __new__(cls, value, **kwargs):
        if isinstance(value, int):
            obj = IntLiteral(value)
        elif isinstance(value, float):
            obj = FloatLiteral(value)
        else:
            # Ensure we capture booleans
            if str(value).lower() in ['.true.', '.false.']:
                value = str(value).strip('.').lower()

            # Let sympy figure out what we're dealing with
            obj = sympy.sympify(value)

            if isinstance(obj, str):
                # Capture strings and ensure they look ok
                obj = String('"%s"' % obj)

        # And attach out own meta-data
        if '_type' in obj.__class__.__slots__:
            obj._type = kwargs.get('type', None)
        if '_kind' in obj.__class__.__slots__:
            obj._kind = kwargs.get('kind', None)
        return obj


class LiteralList(object):

    def __new__(self, values):
        return tuple(Literal(v) for v in values)


class InlineCall(sympy.codegen.ast.FunctionCall, Boolean):
    """
    Internal representation of an in-line function call
    """
    __slots__ = ['name', 'arguments', 'kwarguments']

    defaults = {'arguments': (), 'kwarguments': ()}

    _construct_arguments = staticmethod(lambda args: sympy.Tuple(*args))
    _construct_kwarguments = staticmethod(lambda args: sympy.Tuple(*args))

    def _sympyrepr(self, printer=None):
        """
        Define how we would like to be pretty-printed.
        """
        args = [printer._print(a) for a in self.arguments]
        args += ['%s=%s' % (k, printer._print(v)) for k, v in self.kwarguments]
        return '%s(%s)' % (self.name, ', '.join(args))

    _sympystr = _sympyrepr
    _fcode = _sympyrepr
    _ccode = _sympyrepr

    @property
    def children(self):
        return self.arguments


class Cast(sympy.codegen.ast.FunctionCall):
    """
    Internal representation of a data type cast.
    """
    __slots__ = ['name', 'expression', 'kind']

    defaults = {'kind': sympy.codegen.ast.none}

    @classmethod
    def _construct_kind(cls, argval):
        if argval is None:
            return sympy.codegen.ast.none
        if isinstance(argval, str):
            return sympy.codegen.ast.String(argval)
        else:
            return argval

    def _fcode(self, printer=None):
        """
        Define how we would like to be printed in Fortran code.
        """
        expr = printer._print(self.expression)
        kind = '' if self.kind == None else (', kind=%s' % printer._print(self.kind))
        return '%s(%s%s)' % (self.name, expr, kind)

    _sympyrepr = _fcode
    _sympystr = _sympyrepr


class RangeIndex(sympy.Idx):

    is_Symbol = True
    is_Function = False

    def __new__(cls, *args, **kwargs):
        if len(args) > 0:
            # Already know our symbol
            label = args[0]
            return sympy.Expr.__new__(cls, label)
        else:
            lower = kwargs.get('lower', None)
            upper = kwargs.get('upper', None)
            step = kwargs.get('step', None)

            # Drop trivial default bounds and step sizes
            lower = None if lower == 1 else lower
            step = None if step == 1 else step

            # TODO: Careful, if lower is not None, we get garbage
            # symbol strings (eg. (y, None, 2) => `y:::2`.
            # This is due to a fudge, where we use RangeIndex objects
            # with only upper value as dimension sizes, so they print
            # (None, x, None) => `x`, not `:x` as would be correct.
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
