# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import reduce
import sys

from loki.batch.configure import SchedulerConfig, ItemConfig
from loki.frontend import REGEX, RegexParserClass
from loki.expression import (
    TypedSymbol, MetaSymbol, ProcedureSymbol, Variable
)
from loki.ir import (
    Import, CallStatement, TypeDef, ProcedureDeclaration, Interface,
    FindNodes, FindInlineCalls
)
from loki.module import Module
from loki.subroutine import Subroutine
from loki.tools import as_tuple, flatten, CaseInsensitiveDict
from loki.types import DerivedType


__all__ = [
    'get_all_import_map', 'Item', 'FileItem', 'ModuleItem', 'ProcedureItem',
    'TypeDefItem', 'InterfaceItem', 'ProcedureBindingItem', 'ExternalItem'
]


def get_all_import_map(scope):
    """
    Map of imported symbol names to objects in :data:`scope` and any parent scopes

    For imports that shadow imports in a parent scope, the innermost import
    takes precedence.

    Parameters
    ----------
    scope : :any:`Scope`
        The scope for which the import map is built

    Returns
    -------
    CaseInsensitiveDict
        Mapping of symbol name to symbol object
    """
    imports = getattr(scope, 'imports', ())
    while (scope := scope.parent):
        imports += getattr(scope, 'imports', ())
    return CaseInsensitiveDict(
        (s.name, imprt)
        for imprt in reversed(imports)
        for s in imprt.symbols or [r[1] for r in imprt.rename_list or ()]
    )


class Item(ItemConfig):
    """
    Base class of a work item in the :any:`Scheduler` graph, to which
    a :any:`Transformation` can be applied.

    The :any:`Scheduler` builds a dependency graph consisting of :any:`Item`
    instances as nodes.

    The :attr:`name` of a :class:`Item` refers to the corresponding routine's,
    interface's or type's name using a fully-qualified name in the format
    ``<scope_name>#<local_name>``. The ``<scope_name>`` corresponds to a Fortran
    module that, e.g., a subroutine is declared in, or can be empty if the
    subroutine is not enclosed in a module (i.e. exists in the global scope).
    This is to enable use of routines with the same name that are declared in
    different modules.
    The corresponding parts of the name can be accessed via :attr:`scope_name` and
    :attr:`local_name`.

    For type-bound procedures, the :attr:`local_name` should take the format
    ``<type_name>%<binding_name>``. This may also span across multiple derived types, e.g.,
    to allow calls to type bound procedures of a derived type variable that in turn
    is a member of another derived type, e.g., ``<type_name>%<member_name>%<binding_name>``.
    See :class:`ProcedureBindingItem`` for more details.

    Relation to Loki IR
    -------------------

    Every :any:`Item` corresponds to a specific node in Loki's internal representation.

    For most cases these IR nodes are scopes:

    * :any:`FileItem`: corresponding to :any:`Sourcefile`
    * :any:`ModuleItem`: corresponding to :any:`Module`
    * :any:`ProcedureItem`: corresponding to :any:`Subroutine`

    The remaining cases are items corresponding to IR nodes that constitute some
    form of intermediate dependency, which are required to resolve the indirection
    to the scope node:

    * :any:`InterfaceItem`: corresponding to :any:`Interface`
    * :any:`TypeDefItem`: corresponding to :any:`TypeDef`
    * :any:`ProcedureBindingItem`: corresponding to the :any:`ProcedureSymbol`
      that is declared in a :any:`ProcedureDeclaration` in a derived type.

    The IR node corresponding to an item can be obtain via the :attr:`ir` property.

    Definitions and dependencies of items
    -------------------------------------

    Each item exhibits two key properties:

    * :attr:`definitions`: A list of all IR nodes that constitute symbols/names
    that are made available by an item. For a :any:`FileItem`, this typically consists
    of all modules and procedures in that sourcefile, and for a :any:`ModuleItem` it
    comprises of procedures, interfaces, global variables and derived type definitions.
    * :attr:`dependencies`: A list of all IR nodes that introduce a dependency
    on other items, e.g., :any:`CallStatement` or :any:`Import`.

    Item config
    -----------

    Every item has a bespoke configuration derived from the default values in
    :any:`SchedulerConfig`. The schema and accessible attributes are defined in the
    base class :any:`ItemConfig`.

    Attributes
    ----------

    _parser_class : tuple of :any:`RegexParserClass` or None
        The parser classes that need to be active during a parse with the :any:`REGEX`
        frontend to create the IR nodes corresponding to the item type. This
        class attribute is specified by every class implementing a specific item
        type.
    _defines_items : tuple of subclasses of :any:`Item`
        The types of items that definitions of the item may create. This class
        attribute is specified by every class implementing a specific item type.
    _depends_class : tuple of :any:`RegexParserClass` or None
        The parser classes that need to be active during a parse with the :any:`REGEX`
        frontend to create the IR nodes that constitute dependencies in this
        item type. This class attribute is specified by every class implementing
        a specific item type.
    source : :any:`Sourcefile`
        The sourcefile object in which the IR node corresponding to this item is defined.
        The :attr:`ir` property will look-up and yield the IR node in this source file.
    trafo_data : any:`dict`
        Container object for analysis passes to store analysis data. This can be used
        in subsequent transformation passes.
    plan_data : any:`dict`
        Container object for plan dry-run passes to store information about
        additional and removed dependencies.

    Parameters
    ----------
    name : str
        Name to identify items in the schedulers graph
    source : :any:`Sourcefile`
        The underlying source file that contains the associated item
    config : dict
        Dict of item-specific config options, see :any:`ItemConfig`
    """

    _parser_class = None
    _defines_items = ()
    _depends_class = None

    def __init__(self, name, source, config=None):
        self.name = name
        self.source = source
        self.trafo_data = {}
        self.plan_data = {}
        super().__init__(config)

    def __repr__(self):
        return f'loki.batch.{self.__class__.__name__}<{self.name}>'

    def __eq__(self, other):
        """
        :class:`Item` objects are considered equal if they refer to the same
        fully-qualified name, i.e., :attr:`name` is identical

        This allows also comparison against a string.
        """
        if isinstance(other, Item):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)

    @property
    def definitions(self):
        """
        Return a tuple of the IR nodes this item defines

        By default, this returns an empty tuple and is overwritten by
        derived classes.
        """
        return ()

    @property
    def dependencies(self):
        """
        Return a tuple of IR nodes that constitute dependencies for this item

        This calls :meth:`concretize_dependencies` to trigger a further parse
        with the :any:`REGEX` frontend, including the :attr:`_depends_class` of
        the item. The list of actual dependencies is defined via :meth:`_dependencies`,
        which is overwritten by derived classes.
        """
        self.concretize_dependencies()
        return self._dependencies

    @property
    def _dependencies(self):
        """
        Return a tuple of the IR nodes that constitute dependencies for this item

        This method is used by :attr:`dependencies` to determine the actual
        dependencies after calling :meth:`concretize_dependencies`.

        By default, this returns an empty tuple and is overwritten by
        derived classes.
        """
        return ()

    @property
    def ir(self):
        """
        Return the IR :any:`Node` that the item represents
        """
        return self.source[self.local_name]

    @property
    def scope_ir(self):
        """
        Return the nearest :any:`Scope` IR node that this item either defines
        or is embedded into.
        """
        return self.ir

    def _parser_classes_from_item_type_names(self, item_type_names):
        """
        Helper method that queries the :attr:`Item._parser_class` of all
        :any:`Item` subclasses listed in :data:`item_type_names`
        """
        item_types = [getattr(sys.modules[__name__], name) for name in item_type_names]
        parser_classes = [p for item_type in item_types if (p := item_type._parser_class) is not None]
        return reduce(lambda x, y: x | y, parser_classes, RegexParserClass.EmptyClass)

    def concretize_definitions(self):
        """
        Trigger a re-parse of the source file corresponding to the current item's scope

        This uses :meth:`_parser_classes_from_item_type_names` to determine all
        :any:`RegexParserClass` that the item's definitions require to be parsed.
        An item's definition classes are listed in :attr:`_defines_items`.
        """
        parser_classes = self._parser_classes_from_item_type_names(self._defines_items)
        if parser_classes and hasattr(self.ir, 'make_complete'):
            self.ir.make_complete(frontend=REGEX, parser_classes=parser_classes)

    def concretize_dependencies(self):
        """
        Trigger a re-parse of the source file corresponding to the current item's scope

        This uses :attr:`_depends_class` to determine all :any:`RegexParserClass` that
        the are require to be parsed to find the item's dependencies.
        """
        if not self._depends_class:
            return
        scope = self.scope_ir
        while scope.parent:
            scope = scope.parent
        if hasattr(scope, 'make_complete'):
            scope.make_complete(frontend=REGEX, parser_classes=self._depends_class)

    def create_definition_items(self, item_factory, config=None, only=None):
        """
        Create the :any:`Item` nodes corresponding to the definitions in the
        current item

        Parameters
        ----------
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items
        config : :any:`SchedulerConfig`, optional
            The scheduler config to use when instantiating new items
        only : list of :any:`Item` classes
            Filter the generated items to include only those provided in the list

        Returns
        -------
        tuple
            The list of :any:`Item` nodes
        """
        items = as_tuple(flatten(
            item_factory.create_from_ir(node, self.scope_ir, config)
            for node in self.definitions
        ))
        items = as_tuple(item for item in items if item is not None)
        if only:
            items = tuple(item for item in items if isinstance(item, only))
        return items

    def create_dependency_items(self, item_factory, config=None, only=None):
        """
        Create the :any:`Item` nodes corresponding to the dependencies of the
        current item

        Parameters
        ----------
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items
        config : :any:`SchedulerConfig`, optional
            The scheduler config to use when instantiating new items
        only : list of :any:`Item` classes
            Filter the generated items to include only those provided in the list

        Returns
        -------
        tuple
            The list of :any:`Item` nodes
        """
        ignore = [*self.disable, *self.block]
        items = as_tuple(self.plan_data.get('additional_dependencies'))
        if (dependencies := self.dependencies):
            items += tuple(
                item
                for node in dependencies
                for item in as_tuple(item_factory.create_from_ir(node, self.scope_ir, config, ignore=ignore))
                if item is not None
            )
        if self.disable:
            items = tuple(
                item for item in items
                if not SchedulerConfig.match_item_keys(item.name, self.disable)
            )
        if (removed_dependencies := self.plan_data.get('removed_dependencies')):
            items = tuple(item for item in items if item not in removed_dependencies)

        if only:
            items = tuple(item for item in items if isinstance(item, only))
        return tuple(dict.fromkeys(items))


    @property
    def scope_name(self):
        """
        The name of this item's scope
        """
        pos = self.name.find('#')
        if pos == -1:
            return None
        return self.name[:pos]

    @property
    def local_name(self):
        """
        The item name without the scope
        """
        return self.name[self.name.find('#')+1:]

    @property
    def scope(self):
        """
        IR object that is the enclosing scope of this :any:`Item`

        Note that this property is cached, so that updating the name of an associated
        :any:`Module` (eg. via the :any:`DependencyTransformation`) may not
        break the association with this :any:`Item`.

        Returns
        -------
        :any:`Module` or `NoneType`
        """
        name = self.scope_name
        if name is None:
            return None
        return self.source[name]

    @property
    def calls(self):
        """
        Return a tuple of local names of subroutines that are called

        This will replace the object name by the type name for calls to
        typebound procedures, but not resolve potential renaming via imports.
        """
        calls = tuple(call for call in self.dependencies if isinstance(call, CallStatement))
        calls = tuple(
            f'{call.name.parents[0].type.dtype.name}{call_name[call_name.index("%"):]}'
            if '%' in (call_name := str(call.name).lower()) else call_name
            for call in calls
        )
        return calls

    @property
    def targets(self):
        """
        Set of "active" child dependencies that are part of the transformation
        traversal.

        This includes all child dependencies of an item that will be
        traversed when applying a :any:`Transformation`, after tree pruning rules
        are applied but without taking item filters into account.

        This means, all items excluded via ``block`` or ``disable`` lists in the
        :any:`SchedulerConfig` are not listed here. However, this will include
        items in the ``ignore`` list, which are not processed but are treated
        as if they were.

        Returns
        -------
        list of str
        """
        # Determine an exclusion list
        exclude = as_tuple(str(t).lower() for t in self.disable)
        exclude += as_tuple(str(t).lower() for t in self.block)
        return self._get_children(exclude=exclude)

    @property
    def targets_and_blocked_targets(self):
        """
        Set of all child dependencies, including those that are not part of the
        traversal, but ignoring ``disabled`` dependencies.

        This includes all child dependencies that are returned by :attr:`Item.targets`
        as well as any that are excluded via the :attr:`ItemConfig.block` list.

        This means, only items excluded via ``disable`` lists in the
        :any:`SchedulerConfig` are not listed here. However, it will include
        items in the ``ignore`` and ``block`` list.

        Returns
        -------
        list of str
        """
        # Determine an exclusion list
        exclude = as_tuple(str(t).lower() for t in self.disable)
        return self._get_children(exclude=exclude)

    def _get_children(self, exclude=None):
        """
        Helper method that returns a list of child dependency names

        This takes :attr:`Item.dependencies` and translates the dependency nodes
        to their name, excluding any dependencies that match the exclusion
        list given in :data:`exclude`.

        This method is used by :attr:`targets` and :attr:`targets_and_blocked_targets`.
        """
        exclude = as_tuple(exclude)

        # Determine all potential targets from dependencies and filter out excluded targets
        if not (dependencies := self.dependencies):
            return ()

        def _add_new_child(name, is_excluded, child_exclusion_map):
            # Helper utility to add or update an entry
            child_exclusion_map[name] = child_exclusion_map.get(name, False) or is_excluded

        child_exclusion_map = CaseInsensitiveDict()
        import_map = get_all_import_map(self.scope_ir)
        for dependency in dependencies:
            if isinstance(dependency, Import):
                # Exclude all imported symbols if the module is excluded, otherwise
                # exclude only individual imported symbols as required
                is_excluded = self.match_symbol_or_name(dependency.module, exclude)
                _add_new_child(dependency.module, is_excluded, child_exclusion_map)
                for symbol in dependency.symbols or ():
                    is_symbol_excluded = (
                        is_excluded or symbol.type.parameter or
                        self.match_symbol_or_name(symbol, exclude, scope=dependency.module)
                    )
                    _add_new_child(symbol.name, is_symbol_excluded, child_exclusion_map)

            elif isinstance(dependency, Interface):
                for symbol in dependency.symbols:
                    if symbol.name in import_map:
                        scope = import_map[symbol.name].module
                    else:
                        scope = self.scope_name
                    _add_new_child(
                        symbol.name,
                        self.match_symbol_or_name(symbol, exclude, scope=scope),
                        child_exclusion_map
                    )

            elif isinstance(dependency, TypeDef):
                if dependency.name in import_map:
                    scope = import_map[dependency.name].module
                else:
                    scope = self.scope_name
                _add_new_child(
                    dependency.name,
                    self.match_symbol_or_name(dependency.name, exclude, scope=scope),
                    child_exclusion_map
                )

            elif isinstance(dependency, (Subroutine, CallStatement, MetaSymbol, TypedSymbol)):
                # Treating these together to avoid duplicating the control flow
                # for symbol matching
                if isinstance(dependency, CallStatement):
                    symbol = dependency.name
                elif isinstance(dependency, Subroutine):
                    symbol = dependency.procedure_symbol
                else:
                    symbol = dependency

                if '%' in symbol.name:
                    # We check both:
                    # the (potentially imported) type name via the call relative to
                    # the type name, and the (potentially imported) declared symbol itself
                    type_name = symbol.parents[0].type.dtype.name
                    call_name = f'{type_name}{symbol.name[symbol.name.index("%"):]}'
                    if type_name in import_map:
                        scope = import_map[type_name].module
                    else:
                        scope = self.scope_name
                    is_excluded = self.match_symbol_or_name(call_name, exclude, scope=scope)

                    declared_name = symbol.parents[0].name
                    if (declared_name := symbol.parents[0].name) in import_map:
                        scope = import_map[declared_name].module
                    else:
                        scope = self.scope_name
                    is_excluded = is_excluded or self.match_symbol_or_name(symbol, exclude, scope=scope)

                else:
                    if symbol.name in import_map:
                        scope = import_map[symbol.name].module
                    else:
                        scope = self.scope_name
                    is_excluded = self.match_symbol_or_name(symbol, exclude, scope=scope)

                _add_new_child(symbol.name, is_excluded, child_exclusion_map)
            else:
                raise ValueError(f'Unexpected dependency type {type(dependency)} for {dependency}')

        children = tuple(target for target, excluded in child_exclusion_map.items() if not excluded)
        return children

    @property
    def path(self):
        """
        The filepath of the associated source file
        """
        return self.source.path


class FileItem(Item):
    """
    Item class representing a :any:`Sourcefile`

    The name of this item is typically the file path.

    A :any:`FileItem` does not have any direct dependencies. A dependency
    filegraph can be generated by the :any:`SGraph` class using dependencies
    of items defined by nodes inside the file.

    A :any:`FileItem` defines :any:`ModuleItem` and :any:`ProcedureItem` nodes.
    """

    # We do not need to parse anything inside the file for this item type
    _parser_class = None

    # Modules and Procedures can appear in a sourcefile
    _defines_items = ('ModuleItem', 'ProcedureItem')

    @property
    def definitions(self):
        """
        Return the list of definitions in this source file
        """
        self.concretize_definitions()
        definitions = self.ir.definitions
        for obj in self.ir.definitions:
            if isinstance(obj, Module):
                definitions += obj.definitions
        return self.ir.definitions

    @property
    def ir(self):
        """
        Return the :any:`Sourcefile` associated with this item
        """
        return self.source

    def create_definition_items(self, item_factory, config=None, only=None):
        """
        Create the :any:`Item` nodes corresponding to the definitions in the file

        This overwrites the corresponding method in the base class to enable
        instantiating the top-level scopes in the file item without them being
        available in the :any:`ItemFactory.item_cache`, yet.

        Parameters
        ----------
        item_factory : :any:`ItemFactory`
            The :any:`ItemFactory` to use when creating the items
        config : :any:`SchedulerConfig`, optional
            The scheduler config to use when instantiating new items
        only : list of :any:`Item` classes
            Filter the generated items to include only those provided in the list

        Returns
        -------
        tuple
            The list of :any:`Item` nodes
        """
        items = ()
        for node in self.definitions:
            if isinstance(node, Module):
                items += as_tuple(
                    item_factory.get_or_create_item(ModuleItem, node.name.lower(), self.name, config)
                )
            elif isinstance(node, Subroutine) and not node.parent:
                items += as_tuple(
                    item_factory.get_or_create_item(ProcedureItem, f'#{node.name.lower()}', self.name, config)
                )
            else:
                items += item_factory.create_from_ir(node, self.scope_ir, config)
        items = as_tuple(item for item in items if item is not None)
        if only:
            items = tuple(item for item in items if isinstance(item, only))
        return items


class ModuleItem(Item):
    """
    Item class representing a :any:`Module`

    The name of this item is the module's name, meaning scope name, local name
    and name are all equivalent.

    A :any:`ModuleItem` defines :any:`ProcedureItem`, :any:`InterfaceItem` and
    :any:`TypeDefItem`. Note that global variable imports, which are the fourth
    kind of symbols that can be imported from a module into other scopes are not
    represented by bespoke items.

    A :any:`ModuleItem` can only have a dependency on another :any:`ModuleItem`
    via a :any:`Import` statement.
    """

    _parser_class = RegexParserClass.ProgramUnitClass
    _defines_items = ('ProcedureItem', 'InterfaceItem', 'TypeDefItem')
    _depends_class = RegexParserClass.ImportClass

    @property
    def definitions(self):
        """
        Return the list of definitions in this module, filtering out
        global variables.
        """
        self.concretize_definitions()
        definitions = self.ir.subroutines + as_tuple(FindNodes((TypeDef, Interface)).visit(self.ir.spec))
        return definitions

    @property
    def _dependencies(self):
        """
        Return the list of :any:`Import` nodes that constitute dependencies
        for this module, filtering out imports to intrinsic modules.
        """
        return tuple(
            imprt for imprt in self.ir.imports
            if not imprt.c_import and str(imprt.nature).lower() != 'intrinsic'
        )

    @property
    def local_name(self):
        """
        Return the module's name
        """
        return self.name


class ProcedureItem(Item):
    """
    Item class representing a :any:`Subroutine`

    The name of this item is comprised of the scope's name in which the procedure
    is declared, i.e., the enclosing module, and the procedure name:
    ``<scope_name>#<procedure_name>``. For procedures that are not declared inside
    a module, the ``<scope_name>`` is an empty string, i.e., the item name becomes
    ``#<procedure_name>``.

    A :any:`ProcedureItem` does not define any child items.

    Dependencies of a :any:`ProcedureItem` can be introduced by

    * imports, i.e., a dependency on :any:`ModuleItem`,
    * the use of derived types, i.e., a dependency on :any:`TypeDefItem`,
    * calls to other procedures, i.e., a dependency on :any:`ProcedureItem` or,
      as an indirection, on :any:`InterfaceItem` or :any:`ProcedureBindingItem`.
    """

    _parser_class = RegexParserClass.ProgramUnitClass
    _depends_class = (
        RegexParserClass.ImportClass | RegexParserClass.InterfaceClass | RegexParserClass.TypeDefClass |
        RegexParserClass.DeclarationClass | RegexParserClass.CallClass
    )

    @property
    def _dependencies(self):
        """
        Return the list of :any:`Import`, :any:`Interface`, :any:`TypeDef`,
        :any:`CallStatement`, and :any:`ProcedureSymbol` (to represent
        calls to functions) nodes that constitute dependencies of this item.
        """
        calls = tuple({call.name.name: call for call in FindNodes(CallStatement).visit(self.ir.ir)}.values())
        if internal_procedures := [routine.name.lower() for routine in self.ir.routines]:
            calls = tuple(call for call in calls if call.name.name.lower() not in internal_procedures)
        inline_calls = tuple({
            call.function.name: call.function
            for call in FindInlineCalls().visit(self.ir.ir)
            if isinstance(call.function, ProcedureSymbol) and not call.function.type.is_intrinsic
        }.values())
        imports = tuple(
            imprt for imprt in self.ir.imports
            if not imprt.c_import and str(imprt.nature).lower() != 'intrinsic'
        )
        interfaces = self.ir.interfaces
        typedefs = ()

        # Create dependencies on type definitions that may have been declared in or
        # imported via the module scope
        if self.scope:
            type_names = [
                dtype.name for var in self.ir.variables
                if isinstance((dtype := var.type.dtype), DerivedType)
            ]
            if type_names:
                typedef_map = self.scope.typedef_map
                import_map = self.scope.import_map
                typedefs += tuple(typedef for type_name in type_names if (typedef := typedef_map.get(type_name)))
                imports += tuple(imprt for type_name in type_names if (imprt := import_map.get(type_name)))
        return imports + interfaces + typedefs + calls + inline_calls


class TypeDefItem(Item):
    """
    Item class representing a :any:`TypeDef`

    The name of this item is comprised of the scope's name in which the derived type
    is declared, i.e., the enclosing module, and the type name:
    ``<scope_name>#<type_name>``.

    A :any:`TypeDefItem` defines :any:`ProcedureBindingItem`.

    Dependencies of a :any:`TypeDefItem` are introduced via

    * the use of derived types in declarations of members, i.e., a dependency
      on :any:`TypeDefItem`,
    * imports of derived types, i.e., a dependency on :any:`ModuleItem`.
    """

    _parser_class = RegexParserClass.TypeDefClass
    _defines_items = ('ProcedureBindingItem',)
    _depends_class = RegexParserClass.DeclarationClass

    @property
    def _dependencies(self):
        """
        Return the list of :any:`Import` and :any:`TypeDef` nodes that this item
        depends upon.
        """
        # We restrict dependencies to derived types used in the typedef
        imports = ()
        typedefs = ()

        type_names = [
            dtype.name for var in self.ir.variables
            if isinstance((dtype := var.type.dtype), DerivedType)
        ]
        if type_names:
            typedef_map = self.scope.typedef_map
            import_map = self.scope.import_map
            typedefs = tuple(typedef for type_name in type_names if (typedef := typedef_map.get(type_name)))
            imports = tuple(imprt for type_name in type_names if (imprt := import_map.get(type_name)))

        return tuple(dict.fromkeys(imports + typedefs))

    @property
    def definitions(self):
        """
        Return the list of :any:`ProcedureDeclaration` nodes that define
        procedure bindings in this item.
        """
        return tuple(
            decl for decl in self.ir.declarations
            if isinstance(decl, ProcedureDeclaration)
        )


class InterfaceItem(Item):
    """
    Item class representing a :any:`Interface` declared in a module

    The name of this item is comprised of the scope's name in which the interface
    is declared, i.e., the enclosing module, and the interface name:
    ``<scope_name>#<intf_name>``.

    A :any:`InterfaceItem` does not define any child items.

    The dependency of an :any:`InterfaceItem` is the procedure it declares,
    i.e., a :any:`ProcedureItem` or another :any:`InterfaceItem`.

    This does not constitute a work item when applying transformations across the
    call tree in the :any:`Scheduler` and is skipped by most transformations during
    the processing phase.
    However, it is necessary to provide the dependency link from calls to procedures
    declared via an interface to their implementation in a Fortran routine.
    """

    _parser_class = RegexParserClass.InterfaceClass

    @property
    def _dependencies(self):
        """
        Return the list of :any:`ProcedureSymbol` this interface declares.
        """
        return as_tuple(flatten(
            getattr(node, 'procedure_symbol', getattr(node, 'symbols', ()))
            for node in self.ir.body
        ))

    @property
    def scope_ir(self):
        """
        Return the :any:`Module` in which the interface is declared.
        """
        return self.scope


class ProcedureBindingItem(Item):
    """
    Item class representing a Fortran procedure binding

    The name of this item is comprised of three parts: the scope's name in
    which the derived type with this procedure binding is declared, the name
    of the derived type, and the name of the procedure binding:
    ``<scope_name>#<type_name>%<bind_name>``.

    For nested derived types, the ``<bind_name>`` may consist of multiple parts,
    e.g., ``<scope_name>#<type1_name>%<type1_var>%<type2_var>%<bind_name>``.

    A :any:`ProcedureBindingItem` does not define any child items.

    The dependency of a :any:`ProcedureBindingItem` is the procedure it binds to,
    i.e., a :any:`ProcedureItem`, an :any:`InterfaceItem`, or another
    :any:`ProcedureBindingItem` to resolve generic bindings or calls to bindings
    in nested derived types.

    A :any:`ProcedureBindingItem` does not constitute a work item when applying
    transformations across the dependency tree in the :any:`Scheduler` and is skipped
    during the processing phase by most transformations.
    However, it is necessary to provide the dependency link from calls to type bound
    procedures to their implementation in a Fortran routine.
    """

    _parser_class = RegexParserClass.TypeDefClass | RegexParserClass.CallClass
    _depends_class = RegexParserClass.DeclarationClass

    def __init__(self, name, source, config=None):
        assert '%' in name
        super().__init__(name, source, config)

    @property
    def ir(self):
        """
        Return the :any:`ProcedureSymbol` this binding corresponds to.
        """
        name_parts = self.local_name.split('%')
        typedef = self.source[name_parts[0]]
        if not typedef:
            self.scope.make_complete(frontend=REGEX, parser_classes=self._parser_class)
            typedef = self.source[name_parts[0]]
        for decl in typedef.declarations:
            # We need to compare here explicitly symbol names as the symbol could be
            # declared with a dimension
            for symbol in decl.symbols:
                if name_parts[1] == symbol.name.lower():
                    return decl.symbols[decl.symbols.index(symbol)]
        raise RuntimeError(f'Declaration for {self.name} not found')

    @property
    def scope_ir(self):
        """
        Return the :any:`TypeDef` in which this procedure binding appears.
        """
        return self.ir.scope

    @property
    def _dependencies(self):
        """
        Return the list of :any:`ProcedureSymbol` that correspond to the routine
        binding
        """
        symbol = self.ir
        name_parts = self.local_name.split('%')
        if len(name_parts) == 2:
            if symbol.type.dtype.is_generic:
                # Generic binding
                return tuple(
                    symbol.scope.variable_map[str(name)]
                    for name in as_tuple(symbol.type.bind_names)
                )
            if symbol.type.bind_names:
                # Specific binding with rename
                assert len(symbol.type.bind_names) == 1
                return as_tuple(symbol.type.bind_names[0].type.dtype.procedure)
            return as_tuple(self.source[symbol.name])

        # This is a typebound procedure on a member;
        # let's start by building the (possibly nested) intermediate symbols...
        symbol_name = f'{symbol.name}'
        for name_part in name_parts[2:-1]:
            symbol_name += '%' + name_part
            symbol = Variable(name=symbol_name, parent=symbol, scope=symbol.scope)
        # ...and explicitly instantiate the final symbol as ProcedureSymbol
        proc_name = f'{symbol_name}%{name_parts[-1]}'
        return as_tuple(ProcedureSymbol(name=proc_name, parent=symbol, scope=symbol.scope))


class ExternalItem(Item):
    """
    Item class representing an external dependency that cannot be resolved

    The name of this item may be a fully qualified name containing scope
    and local name, or only a local name.

    It does not define any child items or depend on other items.

    It does not constitute a work item when applying transformations across the
    call tree in the :any:`Scheduler`.

    Parameters
    ----------
    origin_cls :
        The subclass of :any:`Item` this item represents.
    """

    def __init__(self, name, source, config=None, origin_cls=None):
        self.origin_cls = origin_cls
        super().__init__(name, source, config)

    @property
    def ir(self):
        """
        This raises a :any:`RuntimeError`
        """
        raise RuntimeError(f'No .ir available for ExternalItem `{self.name}`')

    @property
    def scope(self):
        """
        This raises a :any:`RuntimeError`
        """
        raise RuntimeError(f'No .scope available for ExternalItem `{self.name}`')

    @property
    def path(self):
        """
        This raises a :any:`RuntimeError`
        """
        raise RuntimeError(f'No .path available for ExternalItem `{self.name}`')
