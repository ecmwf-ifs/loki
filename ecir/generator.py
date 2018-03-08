"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import deque, Iterable, OrderedDict
from itertools import groupby

from ecir.ir import (Loop, Statement, Conditional, Call, Comment, CommentBlock,
                     Pragma, Declaration, Allocation, Variable, Type, DerivedType,
                     Expression, Index, Import, Scope)
from ecir.visitors import Visitor, Transformer, NestedTransformer
from ecir.tools import as_tuple, extract_lines

__all__ = ['generate']


class IRGenerator(Visitor):

    def __init__(self, raw_source):
        super(IRGenerator, self).__init__()

        self._raw_source = raw_source

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        if instance.tag in self._handlers:
            return self._handlers[instance.tag]
        else:
            return super(IRGenerator, self).lookup_method(instance)

    def visit(self, o):
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        try:
            source = extract_lines(o.attrib, self._raw_source)
            line = int(o.attrib['line_begin'])  # Get starting line marker
        except KeyError:
            source = None
            line = None
        return super(IRGenerator, self).visit(o, source=source, line=line)

    def visit_Element(self, o, source=None, line=None):
        """
        Universal default for XML element types
        """
        children = tuple(self.visit(c) for c in o.getchildren())
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        else:
            return children if len(children) > 0 else None

    visit_body = visit_Element

    def visit_loop(self, o, source=None, line=None):
        variable = o.find('header/index-variable').attrib['name']
        try:
            lower = self.visit(o.find('header/index-variable/lower-bound'))
            upper = self.visit(o.find('header/index-variable/upper-bound'))
        except:
            lower = None
            upper = None
        body = as_tuple(self.visit(o.find('body')))
        # Store full lines with loop body for easy replacement
        source = extract_lines(o.attrib, self._raw_source, full_lines=True)
        return Loop(variable=variable, body=body, bounds=(lower, upper),
                    source=source, line=line)

    def visit_if(self, o, source=None, line=None):
        conditions = tuple(self.visit(h) for h in o.findall('header'))
        # HACK around OFP bug (see tools.py:extract_lines):
        # We need t ostrip the additional closing bracked from the
        # expression source.
        for c in conditions:
            if isinstance(c, Expression):
                c.expr = c.expr[:-1]

        bodies = tuple([self.visit(b)] for b in o.findall('body'))
        ncond = len(conditions)
        else_body = bodies[-1] if len(bodies) > ncond else None
        return Conditional(conditions=conditions, bodies=bodies[:ncond],
                           else_body=else_body, source=source, line=line)

    _re_pragma = re.compile('\!\$ecir\s+(?P<keyword>\w+)', re.IGNORECASE)

    def visit_comment(self, o, source=None, line=None):
        match_pragma = self._re_pragma.search(source)
        if match_pragma:
            # Found pragma, generate this instead
            keyword = match_pragma.groupdict()['keyword']
            return Pragma(keyword=keyword, source=source, line=line)
        else:
            return Comment(source=source, line=line)

    def visit_assignment(self, o, source=None, line=None):
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, source=source, line=line)

    def visit_statement(self, o, source=None, line=None):
        if len(o.attrib) == 0:
            return None  # Empty element, skip
        elif o.find('name'):
            # Note: KIND literals confuse the parser, so the structure
            # is slightly odd here. The `name` node is actually the target
            # and the `target` node is actually the KIND identifier.
            target = self.visit(o.find('name'))
            expr = self.visit(o.find('assignment/value'))
            return Statement(target=target, expr=expr, source=source, line=line)
        elif o.find('assignment'):
            return self.visit(o.find('assignment'))
        else:
            return self.visit_Element(o)

    def visit_declaration(self, o, source=None, line=None):
        if len(o.attrib) == 0:
            return None  # Empty element, skip
        elif o.find('save-stmt') is not None:
            return None  # SAVE statement, skip
        elif o.attrib['type'] == 'variable':
            if o.find('end-type-stmt') is not None:
                # We are dealing with a derived type:
                # Things get really messy here, since derived types are
                # (mis-)handled horribly by the OFP: Effectively, each
                # component is hidden recursively in a depth-first
                # hierarchy of 'type' nodes.
                derived_name = o.find('end-type-stmt').attrib['id']
                derived_vars = []
                pragmas = []
                comments = []

                t = o  # We explicitly recurse on t
                while t.find('type') is not None:
                    # Process any associated comments or pragams
                    if t.find('type/comment') is not None:
                        comment = self.visit(t.find('type/comment'))
                        if isinstance(comment, Pragma):
                            pragmas.insert(0, comment)
                        else:
                            comments.insert(0, comment)

                    # Derive type and variables for this entry
                    variables = []
                    attributes = [a.attrib['attrKeyword'] for a in t.findall('component-attr-spec')]
                    typename = t.find('type').attrib['name']  # :(
                    kind = t.find('type/kind/name').attrib['id'] if t.find('type/kind') else None
                    type = Type(typename, kind=kind, pointer='pointer' in attributes)
                    v_source = extract_lines(t.attrib, self._raw_source)
                    v_line = int(t.find('type').attrib['line_end'])
                    for v in t.findall('component-decl'):
                        if 'dimension' in attributes:
                            dim_count = int(t.find('deferred-shape-spec-list').attrib['count'])
                            dimensions = [':' for _ in range(dim_count)]
                        else:
                            dimensions = None
                        variables.append(Variable(name=v.attrib['id'], type=type,
                                                  dimensions=dimensions, source=v_source,
                                                  line=v_line))
                    # Pre-pend current variables to list for this DerivedType
                    derived_vars = variables + derived_vars
                    # Recurse on 'type' nodes
                    t = t.find('type')
                return DerivedType(name=derived_name, variables=derived_vars,
                                   pragmas=pragmas, comments=comments, source=source)
            else:
                # We are dealing with a single declaration, so we retrieve
                # all the declaration-level information first.
                typename = o.find('type').attrib['name']
                kind = o.find('type/kind/name').attrib['id'] if o.find('type/kind') else None
                intent = o.find('intent').attrib['type'] if o.find('intent') else None
                allocatable = o.find('attribute-allocatable') is not None
                optional = o.find('attribute-optional') is not None
                type = Type(name=typename, kind=kind, intent=intent,
                            allocatable=allocatable, optional=optional, source=source)
                variables = []
                for v in o.findall('variables/variable'):
                    if len(v.attrib) == 0:
                        continue
                    dimensions = tuple(self.visit(i) for i in v.findall('dimensions/dimension'))
                    variables.append(Variable(name=v.attrib['name'], type=type,
                                              dimensions=dimensions, source=source))
                return Declaration(variables=variables, source=source, line=line)
        elif o.attrib['type'] == 'implicit':
            return None  # IMPLICIT marker, skip
        else:
            raise NotImplementedError('Unknown declaration type encountered: %s' % o.attrib['type'])

    def visit_associate(self, o, source=None, line=None):
        associations = OrderedDict()
        for a in o.findall('header/keyword-arguments/keyword-argument'):
            var = self.visit(a.find('name'))
            assoc_name = a.find('association').attrib['associate-name']
            associations[var.name] = Variable(name=assoc_name)
        body = self.visit(o.find('body'))
        return Scope(body=body, associations=associations)

    def visit_keyword_argument(self, o, source=None, line=None):
        """Extract a single name => association mapping."""
        return (variable, assoc)

    def visit_allocate(self, o, source=None, line=None):
        variable = self.visit(o.find('expressions/expression/name'))
        return Allocation(variable=variable, source=source, line=line)

    def visit_name(self, o, source=None, line=None):
        indices = tuple(self.visit(i) for i in o.findall('subscripts/subscript'))
        vrefs = o.findall('part-ref')
        vname = '%'.join(i.attrib['id'] for i in vrefs)
        return Variable(name=vname, dimensions=indices)

    def visit_literal(self, o, source=None, line=None):
        return o.attrib['value']

    def visit_value(self, o, source=None, line=None):
        return Expression(source=source)

    def visit_subscript(self, o, source=None, line=None):
        if o.find('range'):
            lower = self.visit(o.find('range/lower-bound'))
            upper = self.visit(o.find('range/upper-bound'))
            return Index(name='%s:%s' % (lower, upper))
        elif o.find('name'):
            # TODO: If the index is a variable,
            # simply return it. This shows that we
            # need a better expression hierachy.
            return self.visit(o.find('name'))
        elif o.find('literal'):
            val = self.visit(o.find('literal'))
            return Index(name='%s' % val)
        elif o.find('operation'):
            op = self.visit(o.find('operation'))
            return Index(name='%s' % op)
        else:
            return Index(name=':')

    def visit_operation(self, o, source=None, line=None):
        return Expression(source=source)

    def visit_use(self, o, source=None, line=None):
        symbols = [n.attrib['id'] for n in o.findall('only/name')]
        return Import(module=o.attrib['name'], symbols=symbols, source=source)

    def visit_call(self, o, source=None, line=None):
        # Need to re-think this: the 'name' node already creates
        # a 'Variable', which in this case is wrong...
        name = o.find('name').attrib['id']
        args = tuple(self.visit(i) for i in o.findall('name/subscripts/subscript'))
        return Call(name=name, arguments=args, source=source, line=line)


class SequenceFinder(Visitor):
    """
    Utility visitor that finds repeated nodes of the same type in
    lists/tuples within a given tree.
    """

    def __init__(self, node_type):
        super(SequenceFinder, self).__init__()
        self.node_type = node_type

    @classmethod
    def default_retval(cls):
        return []

    def visit_tuple(self, o):
        groups = []
        for c in o:
            # First recurse...
            subgroups = self.visit(c)
            if subgroups is not None and len(subgroups) > 0:
                groups += subgroups
        for t, group in groupby(o, lambda o: type(o)):
            # ... then add new groups
            g = tuple(group)
            if t is self.node_type and len(g) > 1:
                groups.append(g)
        return groups

    visit_list = visit_tuple


class PatternFinder(Visitor):
    """
    Utility visitor that finds a pattern of nodes given as tuple/list
    of types within a given tree.
    """

    def __init__(self, pattern):
        super(PatternFinder, self).__init__()
        self.pattern = pattern

    @classmethod
    def default_retval(cls):
        return []

    def match_indices(self, pattern, sequence):
        """ Return indices of matched patterns in sequence. """
        matches = []
        for i in range(len(sequence)):
            if sequence[i] == pattern[0]:
                if tuple(sequence[i:i+len(pattern)]) == tuple(pattern):
                    matches.append(i)
        return matches

    def visit_tuple(self, o):
        matches = []
        for c in o:
            # First recurse...
            submatches = self.visit(c)
            if submatches is not None and len(submatches) > 0:
                matches += submatches
        types = list(map(type, o))
        idx = self.match_indices(self.pattern, types)
        for i in idx:
            matches.append(o[i:i+len(self.pattern)])
        return matches

    visit_list = visit_tuple


def generate(ofp_ast, raw_source):
    """
    Generate an internal IR from the raw OFP (Open Fortran Parser)
    output.

    The internal IR is intended to represent the code at a much higher
    level than the raw langage constructs that OFP returns.
    """

    # Parse the OFP AST into a raw IR
    ir = IRGenerator(raw_source).visit(ofp_ast)

    # Identify inline comments and merge them onto statements
    pairs = PatternFinder(pattern=(Statement, Comment)).visit(ir)
    pairs += PatternFinder(pattern=(Declaration, Comment)).visit(ir)
    mapper = {}
    for pair in pairs:
        if pair[0]._line == pair[1]._line:
            # Comment is in-line and can be merged
            # Note, we need to re-create the statement node
            # so that Transformers don't throw away the changes.
            mapper[pair[0]] = pair[0]._rebuild(comment=pair[1])
            mapper[pair[1]] = None  # Mark for deletion
    ir = NestedTransformer(mapper).visit(ir)

    # Cluster comments into comment blocks
    comment_mapper = {}
    comment_groups = SequenceFinder(node_type=Comment).visit(ir)
    for comments in comment_groups:
        # Build a CommentBlock and map it to first comment
        # and map remaining comments to None for removal
        block = CommentBlock(comments)
        comment_mapper[comments[0]] = block
        for c in comments[1:]:
            comment_mapper[c] = None
    ir = NestedTransformer(comment_mapper).visit(ir)

    # Find pragmas and merge them onto declarations and subroutine calls
    # Note: Pragmas in derived types are already associated with the
    # declaration due to way we parse derived types.
    pairs = PatternFinder(pattern=(Pragma, Declaration)).visit(ir)
    pairs += PatternFinder(pattern=(Pragma, Call)).visit(ir)
    mapper = {}
    for pair in pairs:
        if pair[0]._line == pair[1]._line - 1:
            # Merge pragma with declaration and delete
            mapper[pair[0]] = None  # Mark for deletion
            mapper[pair[1]] = pair[1]._rebuild(pragma=pair[0])
    ir = NestedTransformer(mapper).visit(ir)

    return ir
