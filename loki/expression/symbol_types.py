import sympy
import weakref
from sympy.core.cache import cacheit, SYMPY_CACHE_SIZE
from sympy.logic.boolalg import Boolean
from sympy.codegen.ast import String, Pointer, none
from sympy.codegen.fnodes import ArrayConstructor
from sympy.printing.codeprinter import CodePrinter
from fastcache import clru_cache

from loki.types import DataType


__all__ = ['Scalar', 'Array', 'Variable', 'Literal', 'LiteralList',
           'RangeIndex', 'InlineCall', 'Cast', 'indexify', 'SymbolCache']


def _symbol_class(cls, name, parent=None, cache=None):
    """
    Create new type instance from cls and inject symbol name

    Note, in order to keep multiple independent caches, play nicely
    with SymPys reinstaniation mechanism and restore meta-data from
    our caches, we need to hide a reference to our original cache in
    the class definition.
    """
    # Add the parent-object if it exists (`parent`)
    parent = ('%s%%' % parent) if parent is not None else ''
    name = '%s%s' % (parent, name)

    newcls = type(name, (cls, ), dict(cls.__dict__))
    if cache is not None:
        newcls._cache = weakref.ref(cache)
    return newcls


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
_global_class_cache = cacheit(_symbol_class)
_global_meta_cache = {}


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
        self._class_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                       unhashable='ignore')(_symbol_class)
        self._array_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                       unhashable='ignore')(Array.__new_stage2__)
        self._scalar_cache = clru_cache(SYMPY_CACHE_SIZE, typed=True,
                                        unhashable='ignore')(Scalar.__new_stage2__)
        self._meta_cache = {}

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
        # Shape is not used to cache, so we leave it
        # in kwargs for __init__ to fully use it
        shape = kwargs.get('shape', None)

        # Create scalar or array symbol using local caches
        if dimensions is None and (shape is None or len(shape) == 0):
            cls = self._class_cache(Scalar, name, parent, cache=self)
            newobj = self._scalar_cache(cls, name, parent=parent)
        else:
            _type = kwargs.get('type', None)
            dimensions = () if dimensions is None else dimensions
            if _type and _type.dtype == DataType.BOOL:
                cls = self._class_cache(BoolArray, name, parent, cache=self)
            else:
                cls = self._class_cache(Array, name, parent, cache=self)
            newobj = self._array_cache(cls, name, dimensions=dimensions, parent=parent)

        # Since we are not actually using the object instation
        # mechanism, we need to call __init__ ourselves.
        newobj.__init__(*args, **kwargs)
        return newobj


class CachedMeta(object):
    """
    Base class for addtional meta-data caching on top of symbol caching.

    This class provides the mechanism that re-attaches instance variable
    """

    def __init__(self, *args, **kwargs):
        if hasattr(self.__class__, '_cache'):
            # Pull our original cache from the symbol __class__
            cache = self.__class__._cache()._meta_cache
        else:
            cache = _global_meta_cache

        # Second-level caching of meta-date info
        if self.__class__ in cache:
            self.__dict__ = cache[self.__class__]().__dict__
        else:
            cache[self.__class__] = weakref.ref(self)


class Scalar(sympy.Symbol, CachedMeta):

    is_Scalar = True
    is_Array = False

    def __new__(cls, *args, **kwargs):
        """
        1st-level variable creation with name injection via the object class
        """
        name = kwargs.pop('name', args[0] if len(args) > 0 else None)
        parent = kwargs.pop('parent', None)
        # Name injection for symbols via cls (so we can do `a%scalar`)
        cls = _global_class_cache(Scalar, name, parent)

        # Create a new object from the static constructor with global caching!
        return Scalar.__xnew_cached_(cls, name, parent=parent)

    def __new_stage2__(cls, name, parent=None):
        """
        2nd-level constructor: arguments to this constructor are used
        for symbolic caching
        """
        # Setting things here before __init__ forces them
        # to be used for symbolic caching. Thus, `parent` is
        # always used for caching, even if it's not in the name
        newobj = sympy.Symbol.__new__(cls, name)
        # Use cls.__name__ to get the injected name modifcation
        newobj.name = cls.__name__
        newobj.basename = name
        newobj.parent = parent

        return newobj

    # Create a globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __attr_from_kwargs(self, attr, kwargs, key=None):
        """
        Helper to set object attributes during ``symbol.__init__(...)``
        that prevents accidental overriding of previously cached values.
        """
        key = attr if key is None else key

        # Ensure attribute exists
        if not hasattr(self, attr):
            setattr(self, attr, None)

        # Pull value from kwargs if it's not None
        if key in kwargs and kwargs[key] is not None:
            setattr(self, attr, kwargs[key])

    def __init__(self, *args, **kwargs):
        """
        Initialisation of non-cached objects attributes
        """
        # Initialize meta attributes from cache
        super(Scalar, self).__init__(*args, **kwargs)

        # Ensure all non-cached attributes exists and override if
        # explicitly provided. Is there a nicer way to do this?
        self.__attr_from_kwargs('_source', kwargs, key='_source')
        self.__attr_from_kwargs('initial', kwargs, key='initial')
        self.__attr_from_kwargs('_type', kwargs, key='type')

    def clone(self, **kwargs):
        """
        Replicate the :class:`Scalar` variable with the provided overrides.

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
        if cache is None and hasattr(self.__class__, '_cache'):
            cache = self.__class__._cache()
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

    def _ccode(self, printer=None):
        s = CodePrinter._print_Symbol(printer, self)
        if self.type is not None and self.type.pointer:
            s = '*%s' % s
        return s.replace('%', '->')


class Array(sympy.Function, CachedMeta):

    is_Scalar = False
    is_Array = True

    # Loki array variable are by default non-commutative to preserve
    # term ordering in defensive "parse-unparse" cycles.
    is_commutative = False

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
                cls = _global_class_cache(BoolArray, name, parent)
            else:
                cls = _global_class_cache(Array, name, parent)
        else:
            # A reconstruction of an array(function) object,
            # as triggered during symbolic manipulation.
            name = cls.__name__
            dimensions = args
            parent = kwargs.pop('parent', None)

        # Create a new object from the static constructor with global caching!
        return Array.__xnew_cached_(cls, name, dimensions=dimensions, parent=parent)

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
        newobj.parent = parent

        return newobj

    # Use the sympy.core.cache.cacheit decorator to a kernel to create
    # a static globally cached symbol constructor.
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __attr_from_kwargs(self, attr, kwargs, key=None):
        """
        Helper to set object attributes during ``symbol.__init__(...)``
        that prevents accidental overriding of previously cached values.
        """
        key = attr if key is None else key

        # Ensure attribute exists
        if not hasattr(self, attr):
            setattr(self, attr, None)

        # Pull value from kwargs if it's not None
        if key in kwargs and kwargs[key] is not None:
            setattr(self, attr, kwargs[key])

    def __init__(self, *args, **kwargs):
        """
        Initialisation of non-cached objects attributes
        """
        # Initialize meta attributes from cache
        super(Array, self).__init__(*args, **kwargs)

        # Ensure all non-cached attributes exists and override if
        # explicitly provided. Is there a nicer way to do this?
        self.__attr_from_kwargs('_source', kwargs, key='_source')
        self.__attr_from_kwargs('initial', kwargs, key='initial')
        self.__attr_from_kwargs('_type', kwargs, key='type')
        self.__attr_from_kwargs('_shape', kwargs, key='shape')

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
        if cache is None and hasattr(self.__class__, '_cache'):
            cache = self.__class__._cache()

        if cache is None:
            return Variable(**kwargs)
        else:
            return cache.Variable(**kwargs)

    @property
    def dimensions(self):
        """
        Symbolic representation of the dimensions or indices.
        """
        return self.args

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

    def _sympyrepr(self, printer=None):
        """
        Define how we would like to be printed in Fortran code. This will
        remove any indices/sizes if ``len(self.dimensions) == 0``.

        We need to override this, so that we may have :class:`sympy.Function`
        behaviour, but allow empty ``.args`` tuples for representing
        arrays without indices/sizes (for example in a function call).
        """
        s = self.func.__name__
        if len(self.args) > 0:
            a = ','.join(printer.doprint(a) for a in self.args)
            s += '(%s)' % a
        return s

    _sympystr = _sympyrepr
    _fcode = _sympyrepr

    @property
    def indexed(self):
        return sympy.IndexedBase(self.name, shape=self.shape or self.args)

    def indexify(self):
        """
        Return an instance of :class:`sympy.Indexed` that corresponds
        to this :class:`Array` variable. This is useful for generating
        code for array accesses using the sympy code printers.
        """
        try:
            return self.indexed[self.args]
        except sympy.indexed.IndexException:
            return Pointer(self)


class BoolArray(Array, Boolean):
    """
    A specialised form of an :class:`Array` variable that behaves like
    a :class:`sympy.logic.boolargl.Boolean`, allowing us to use it in
    logical expressions.
    """

    is_Boolean = True
    # Boolean (lattice) expressions cancel commutative terms,
    # so we need to revert our hack here to prevent side-effects.
    is_commutative = True

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

    is_Boolean = True
    # Boolean (lattice) expressions cancel commutative terms,
    # so we need to revert our hack here to prevent side-effects.
    is_commutative = True

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
        # Shape and type are not used for caching, so we leave it in
        # kwargs for __init__ to take it off
        _type = kwargs.get('type', None)
        shape = kwargs.get('shape', None)

        # Create a new object from the static constructor with global caching!
        if dimensions is None and (shape is None or len(shape) == 0):
            newobj = Scalar.__new__(Scalar, name=name, parent=parent)
        else:
            newobj = Array.__new__(Array, name=name, dimensions=dimensions, parent=parent, type=_type)

        # Explicit invocation of __init__ for non-sympy meta data
        newobj.__init__(*args, **kwargs)
        return newobj


class FloatLiteral(sympy.Float):
    __slots__ = ['_mpf_', '_prec', '_type', '_kind']

    def _fcode(self, printer=None):
        printed = CodePrinter._print_Float(printer, self)
        if hasattr(self, '_kind') and self._kind is not None:
            return '%s_%s' % (printed, self._kind)
        else:
            return printed

    def _ccode(self, printer=None):
        return CodePrinter._print_Float(printer, self)


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
        return ArrayConstructor(elements=[Literal(v) for v in values])


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

    def _ccode(self, printer=None):
        """
        Define how we would like to be printed in Fortrano code.
        """
        expr = printer._print(self.expression)
        # TODO: Until we sort out or types, we simply derive it
        dtype = DataType.from_type_kind(self.name, self.kind)
        return '(%s)(%s)' % (dtype.ctype, expr)


class RangeIndex(sympy.Idx):

    is_Symbol = True
    is_Function = False
    is_Scalar = False
    is_Array = False

    @classmethod
    def _label(cls, lower, upper, step):
        upper = None if upper is none else upper
        lower = None if lower is none else lower
        step = None if step is none else step

        upper = '' if upper is None else str(upper)
        lower = '' if lower is None else str(lower)
        label = '%s:%s' % (lower, upper)
        if step is not None:
            label = '%s:%s' % (label, step)
        return label

    @cacheit
    def __new__(cls, *args, **kwargs):
        lower, upper, step = None, None, None
        if len(args) == 1:
            upper = args[0]
        elif len(args) == 2:
            lower = args[0]
            upper = args[1]
        elif len(args) == 3:
            lower = args[0]
            upper = args[1]
            step = args[2]

        lower = kwargs.get('lower', lower)
        upper = kwargs.get('upper', upper)
        step = kwargs.get('step', step)

        # Short-circuit for direct indices
        if upper is not None and lower is None and step is None:
            return upper

        label = cls._label(lower, upper, step)

        obj = sympy.Expr.__new__(cls, label)
        obj._lower = lower
        obj._upper = upper
        obj._step = step

        return obj

    @property
    def args(self):
        args = []
        args += [none] if self._lower is None else [self._lower]
        args += [none] if self._upper is None else [self._upper]
        args += [none] if self._step is None else [self._step]
        return args

    def _sympystr(self, p):
        return self._label(self._lower, self._upper, self._step)

    _sympyrepr = _sympystr
    _fcode = _sympystr

    def _ccode(self, p):
        return ''

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def step(self):
        return self._step
