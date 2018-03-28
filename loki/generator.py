"""
Module to manage loops and statements via an internal representation(IR)/AST.
"""

import re
from collections import OrderedDict, deque
from itertools import groupby

from loki.ir import (Loop, Statement, Conditional, Call, Comment, CommentBlock,
                     Pragma, Declaration, Allocation, Deallocation, Nullify,
                     Import, Scope, Intrinsic, TypeDef, MaskedStatement)
from loki.expression import (Variable, Literal, Operation, Index, InlineCall)
from loki.types import BaseType
from loki.visitors import GenericVisitor, Visitor, NestedTransformer
from loki.tools import as_tuple, timeit

__all__ = ['generate', 'extract_source', 'Source']


class Source(object):

    def __init__(self, string, lines):
        self.string = string
        self.lines = lines

    def __repr__(self):
        return 'Source<lines %s-%s>' % (self.lines[0], self.lines[1])


def extract_source(ast, text, full_lines=False):
    """
    Extract the marked string from source text.
    """
    attrib = ast.attrib if hasattr(ast, 'attrib') else ast
    lstart = int(attrib['line_begin'])
    lend = int(attrib['line_end'])
    cstart = int(attrib['col_begin'])
    cend = int(attrib['col_end'])

    text = text.splitlines(keepends=True)

    if full_lines:
        return Source(string=''.join(text[lstart-1:lend]),
                      lines=(lstart, lend))

    lines = text[lstart-1:lend]

    # Scan for line continuations and honour inline
    # comments in between continued lines
    def continued(line):
        return line.strip().endswith('&')

    def is_comment(line):
        return line.strip().startswith('!')

    # We only honour line continuation if we're not parsing a comment
    if not is_comment(lines[-1]):
        while continued(lines[-1]) or is_comment(lines[-1]):
            lend += 1
            # TODO: Strip the leading empty space before the '&'
            lines.append(text[lend-1])

    # If line continuation is used, move column index to the relevant parts
    while cstart >= len(lines[0]):
        if not is_comment(lines[0]):
            cstart -= len(lines[0])
            cend -= len(lines[0])
        lines = lines[1:]
        lstart += 1

    # TODO: The column indexes are still not right, so source strings
    # for sub-expressions are likely wrong!
    if lstart == lend:
        lines[0] = lines[0][cstart:cend]
    else:
        lines[0] = lines[0][cstart:]
        lines[-1] = lines[-1][:cend]

    return Source(string=''.join(lines), lines=(lstart, lend))


class XMLVisitor(GenericVisitor):
    """
    Base class for XML visitors that traverse XML node trees.

    Note: Since re-building these trees is insane, we will modify them
    in-place and use these visitor classes as "transformers" as well.
    """

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        else:
            return super(XMLVisitor, self).lookup_method(instance)

    def visit_Element(self, o):
        for e in list(o):
            self.visit(e)


class OFPDropEmptyRules(XMLVisitor):
    """
    Remove all elements that only contain rule attributes
    """

    def visit_Element(self, o):
        for e in list(o):
            if len(list(e)) == 0 and len(e.attrib) == 1 and 'rule' in e.attrib:
                o.remove(e)
            else:
                self.visit(e)


class OFPFixMultipartVariables(XMLVisitor):
    """
    Explicit OFP bugfix for erroneously nested keywrod argument lists:

    If derived-type indirections deeper than 2 levels (eg. A%B%C) are
    present in keyword-argument lists or subscript lists, the OFP will
    erroneously split the compound 'name' node and hoist some 'part-ref'
    nodes above the containing 'keyword-argument' node. This creates a
    weird nested structure that we flatten here by re-connecting
    the XML nodes in-place.
    """

    def merge(self, head, tail, parent):
        """
        Utility to merge two 'name' nodes, where the 'part-ref' nodes
        in `head` are inserted at the front of `tail`, before removing
        the `head` and hoisting all its children to the `parent`.
        """
        head_refs = head.findall('part-ref')
        tail_refs = tail.findall('part-ref')

        # Shove head parts into the tail
        for ref in head_refs:
            tail.insert(0, ref)
            head.remove(ref)

        # Drop the now empty head and hoist its children
        parent.remove(head)
        for c in list(head):
            # parent.insert(-1, c)
            parent.append(c)

    def merge_multipart(self, head, parent):
        """
        Utility to recursively descend into the last layer of nested
        'name' node pairs, before merging our way back up.
        """
        if head.find('name'):
            self.merge_multipart(head=head.find('name'), parent=head)

        tail = head.find('*/name')
        if tail is not None:
            self.merge(head, tail, parent=parent)

    def visit_keyword_arguments(self, o):
        while o.find('name') is not None:
            head = o.find('name')
            self.merge_multipart(head=head, parent=o)

    def visit_call(self, o):
        subscripts = o.find('name/subscripts')
        while subscripts.find('name') is not None:
            head = subscripts.find('name')
            self.merge_multipart(head=head, parent=subscripts)


class IRGenerator(GenericVisitor):

    def __init__(self, raw_source):
        super(IRGenerator, self).__init__()

        self._raw_source = raw_source

    def lookup_method(self, instance):
        """
        Alternative lookup method for XML element types, identified by ``element.tag``
        """
        tag = instance.tag.replace('-', '_')
        if tag in self._handlers:
            return self._handlers[tag]
        else:
            return super(IRGenerator, self).lookup_method(instance)

    def visit(self, o):
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        try:
            source = extract_source(o.attrib, self._raw_source)
        except KeyError:
            source = None
        return super(IRGenerator, self).visit(o, source=source)

    def visit_Element(self, o, source=None):
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

    def visit_loop(self, o, source=None):
        variable = o.find('header/index-variable').attrib['name']

        lower = self.visit(o.find('header/index-variable/lower-bound'))
        upper = self.visit(o.find('header/index-variable/upper-bound'))
        step = None
        if o.find('header/index-variable/step') is not None:
            step = self.visit(o.find('header/index-variable/step'))

        body = as_tuple(self.visit(o.find('body')))
        # Store full lines with loop body for easy replacement
        source = extract_source(o.attrib, self._raw_source, full_lines=True)
        return Loop(variable=variable, body=body, bounds=(lower, upper, step),
                    source=source)

    def visit_if(self, o, source=None):
        conditions = tuple(self.visit(h) for h in o.findall('header'))
        bodies = tuple([self.visit(b)] for b in o.findall('body'))
        ncond = len(conditions)
        else_body = bodies[-1] if len(bodies) > ncond else None
        return Conditional(conditions=conditions, bodies=bodies[:ncond],
                           else_body=else_body, source=source)

    _re_pragma = re.compile('\!\$ecir\s+(?P<keyword>\w+)', re.IGNORECASE)

    def visit_comment(self, o, source=None):
        match_pragma = self._re_pragma.search(source.string)
        if match_pragma:
            # Found pragma, generate this instead
            keyword = match_pragma.groupdict()['keyword']
            return Pragma(keyword=keyword, source=source)
        else:
            return Comment(source=source)

    def visit_statement(self, o, source=None):
        # TODO: Hacky pre-emption for special-case statements
        if o.find('name/nullify-stmt') is not None:
            variable = self.visit(o.find('name'))
            return Nullify(variable=variable, source=source)
        elif o.find('where-construct-stmt') is not None:
            # Parse a WHERE statement...
            condition = self.visit(o[0])
            body = self.visit(o[2])
            default = None
            for i in range(len(o)):
                if o[i].tag == 'elsewhere-stmt':
                    default = self.visit(o[i+1])
            # TODO: Deal with alternative conditions (multiple ELSEWHERE)
            return MaskedStatement(condition=condition, body=body, default=default)
        else:
            return self.visit_Element(o, source=source)

    def visit_assignment(self, o, source=None):
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, source=source)

    def visit_pointer_assignment(self, o, source=None):
        target = self.visit(o.find('target'))
        expr = self.visit(o.find('value'))
        return Statement(target=target, expr=expr, ptr=True, source=source)

    def visit_declaration(self, o, source=None):
        if len(o.attrib) == 0:
            return None  # Empty element, skip
        elif o.find('save-stmt') is not None:
            # SAVE statement
            return Intrinsic(source=source)
        elif o.attrib['type'] == 'variable':
            if o.find('end-type-stmt') is not None:
                # We are dealing with a derived type:
                # Things get really messy here, since derived types are
                # (mis-)handled horribly by the OFP: Effectively, each
                # component is hidden recursively in a depth-first
                # hierarchy of 'type' nodes.
                derived_name = o.find('end-type-stmt').attrib['id']
                declarations = []
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
                    attributes = [a.attrib['attrKeyword'].upper()
                                  for a in t.findall('component-attr-spec')]
                    typename = t.find('type').attrib['name']  # :(
                    kind = t.find('type/kind/name').attrib['id'] if t.find('type/kind') else None
                    type = BaseType(typename, kind=kind, pointer='POINTER' in attributes)
                    v_source = extract_source(t.attrib, self._raw_source)
                    v_line = int(t.find('type').attrib['line_end'])
                    v_source.lines = (v_line, v_line)  # HACK!!!
                    for v in t.findall('component-decl'):
                        deferred_shape = t.find('deferred-shape-spec-list')
                        if deferred_shape is not None:
                            dim_count = int(deferred_shape.attrib['count'])
                            dimensions = [':' for _ in range(dim_count)]
                        else:
                            dimensions = None
                        variables.append(Variable(name=v.attrib['id'], type=type,
                                                  dimensions=dimensions, source=v_source))
                    # Pre-pend current variables to list for this DerivedType
                    declarations.insert(0, Declaration(variables=variables))
                    # Recurse on 'type' nodes
                    t = t.find('type')
                return TypeDef(name=derived_name, declarations=declarations,
                               pragmas=pragmas, comments=comments, source=source)
            else:
                # We are dealing with a single declaration, so we retrieve
                # all the declaration-level information first.
                typename = o.find('type').attrib['name']
                kind = o.find('type/kind/name').attrib['id'] if o.find('type/kind') else None
                intent = o.find('intent').attrib['type'] if o.find('intent') else None
                allocatable = o.find('attribute-allocatable') is not None
                pointer = o.find('attribute-pointer') is not None
                parameter = o.find('attribute-parameter') is not None
                optional = o.find('attribute-optional') is not None
                target = o.find('attribute-target') is not None
                type = BaseType(name=typename, kind=kind, intent=intent,
                                allocatable=allocatable, pointer=pointer,
                                optional=optional, parameter=parameter,
                                target=target, source=source)
                variables = [self.visit(v) for v in o.findall('variables/variable')]
                variables = [v for v in variables if v is not None]
                # Retro-fit type onto variables
                for v in variables:
                    v._type = type
                return Declaration(variables=variables, source=source)
        elif o.attrib['type'] == 'implicit':
            # IMPLICIT marker
            return Intrinsic(source=source)
        else:
            raise NotImplementedError('Unknown declaration type encountered: %s' % o.attrib['type'])

    def visit_associate(self, o, source=None):
        associations = OrderedDict()
        for a in o.findall('header/keyword-arguments/keyword-argument'):
            var = self.visit(a.find('name'))
            assoc_name = a.find('association').attrib['associate-name']
            associations[var] = Variable(name=assoc_name)
        body = self.visit(o.find('body'))
        return Scope(body=body, associations=associations)

    def visit_allocate(self, o, source=None):
        variable = self.visit(o.find('expressions/expression/name'))
        return Allocation(variable=variable, source=source)

    def visit_deallocate(self, o, source=None):
        variable = self.visit(o.find('expressions/expression/name'))
        return Deallocation(variable=variable, source=source)

    def visit_use(self, o, source=None):
        symbols = [n.attrib['id'] for n in o.findall('only/name')]
        return Import(module=o.attrib['name'], symbols=symbols, source=source)

    def visit_directive(self, o, source=None):
        # Straight pipe-through node for header includes (#include ...)
        return Intrinsic(source=source)

    def visit_open(self, o, source=None):
        return Intrinsic(source=source)

    visit_close = visit_open
    visit_read = visit_open
    visit_write = visit_open
    visit_format = visit_open

    def visit_call(self, o, source=None):
        # Need to re-think this: the 'name' node already creates
        # a 'Variable', which in this case is wrong...
        name = o.find('name').attrib['id']
        args = tuple(self.visit(i) for i in o.findall('name/subscripts/subscript'))
        kwargs = OrderedDict([self.visit(i) for i in o.findall('name/subscripts/argument')])
        return Call(name=name, arguments=args, kwarguments=kwargs, source=source)

    def visit_argument(self, o, source=None):
        key = o.attrib['name']
        val = self.visit(o.find('name'))
        return key, val

    # Expression parsing below; maye move to its own parser..?

    def visit_name(self, o, source=None):

        def generate_variable(vname, indices, subvar, source):
            if vname.upper() in ['MIN', 'MAX', 'EXP', 'SQRT', 'ABS']:
                return InlineCall(name=vname, arguments=indices)
            elif indices is not None and len(indices) == 0:
                # HACK: We (most likely) found a call out to a C routine
                return InlineCall(name=o.attrib['id'], arguments=indices)
            else:
                return Variable(name=vname, dimensions=indices,
                                subvar=variable, source=source)

        # Creating compound variables is a bit tricky, so let's first
        # process all our children and shove them into a deque
        _children = deque(self.visit(c) for c in o.getchildren())
        _children = deque(c for c in _children if c is not None)

        # Now we nest variables, dimensions and sub-variables by
        # popping them off the back of our deque...
        indices = None
        variable = None
        while len(_children) > 0:
            item = _children.pop()
            if len(_children) > 0 and isinstance(_children[-1], tuple):
                indices = _children.pop()

            if len(_children) > 0 and isinstance(item, Variable):
                # A subvar was processed separately, so move over
                if variable is None:
                    variable = item
                    continue

            # The "append" base case
            variable = generate_variable(vname=item, indices=indices,
                                         subvar=variable, source=source)
            indices = None
        return variable

    def visit_variable(self, o, source=None):
        if 'id' not in o.attrib and 'name' not in o.attrib:
            return None
        name = o.attrib['id'] if 'id' in o.attrib else o.attrib['name']
        if o.find('dimensions') is not None:
            dimensions = tuple(self.visit(d) for d in o.find('dimensions'))
            dimensions = tuple(d for d in dimensions if d is not None)
        else:
            dimensions = None
        initial = None if o.find('initial-value') is None else self.visit(o.find('initial-value'))
        return Variable(name=name, dimensions=dimensions, initial=initial, source=source)

    def visit_part_ref(self, o, source=None):
        # Return a pure string, as part of a variable name
        return o.attrib['id']

    def visit_literal(self, o, source=None):
        value = o.attrib['value']
        # Override Fortran BOOL keywords
        if value == 'false':
            value = '.FALSE.'
        if value == 'true':
            value = '.TRUE.'
        return Literal(value=value, source=source)

    def visit_subscripts(self, o, source=None):
        return tuple(self.visit(c)for c in o.getchildren()
                     if c.tag in ['subscript', 'name'])

    def visit_subscript(self, o, source=None):
        if o.find('range'):
            lower = self.visit(o.find('range/lower-bound'))
            upper = self.visit(o.find('range/upper-bound'))
            return Index(name='%s:%s' % (lower, upper))
        elif o.find('name'):
            return self.visit(o.find('name'))
        elif o.find('literal'):
            return self.visit(o.find('literal'))
        elif o.find('operation'):
            return self.visit(o.find('operation'))
        else:
            return Index(name=':')

    visit_dimension = visit_subscript

    def visit_operation(self, o, source=None):
        ops = [self.visit(op) for op in o.findall('operator')]
        ops = [op for op in ops if op is not None]  # Filter empty ops
        exprs = [self.visit(c) for c in o.findall('operand')]
        exprs = [e for e in exprs if e is not None]  # Filter empty operands
        parenthesis = o.find('parenthesized_expr') is not None

        return Operation(ops=ops, operands=exprs, parenthesis=parenthesis,
                         source=source)

    def visit_operator(self, o, source=None):
        return o.attrib['operator']


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


@timeit()
def generate(ofp_ast, raw_source):
    """
    Generate an internal IR from the raw OFP (Open Fortran Parser)
    output.

    The internal IR is intended to represent the code at a much higher
    level than the raw langage constructs that OFP returns.
    """

    # Sanitize the OFP output in-place
    OFPDropEmptyRules().visit(ofp_ast)
    OFPFixMultipartVariables().visit(ofp_ast)

    # Parse the OFP AST into a raw IR
    ir = IRGenerator(raw_source).visit(ofp_ast)

    # Identify inline comments and merge them onto statements
    pairs = PatternFinder(pattern=(Statement, Comment)).visit(ir)
    pairs += PatternFinder(pattern=(Declaration, Comment)).visit(ir)
    mapper = {}
    for pair in pairs:
        if pair[0]._source.lines[0] == pair[1]._source.lines[0]:
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
        if pair[0]._source.lines[0] == pair[1]._source.lines[0] - 1:
            # Merge pragma with declaration and delete
            mapper[pair[0]] = None  # Mark for deletion
            mapper[pair[1]] = pair[1]._rebuild(pragma=pair[0])
    ir = NestedTransformer(mapper).visit(ir)

    return ir
