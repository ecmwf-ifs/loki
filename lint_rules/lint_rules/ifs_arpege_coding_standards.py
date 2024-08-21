# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Implementation of rules from the IFS Arpege coding standards as :any:`GenericRule`

See https://sites.ecmwf.int/docs/ifs-arpege-coding-standards/fortran for the
current version of the coding standards.
"""

from collections import defaultdict
import re
import difflib

try:
    from fparser.two.Fortran2003 import Intrinsic_Name
    _intrinsic_fortran_names = Intrinsic_Name.function_names
except ImportError:
    _intrinsic_fortran_names = ()

from loki import (
    FindInlineCalls, FindNodes, GenericRule, Module, RuleType,
    ExpressionFinder, ExpressionRetriever, FloatLiteral,
    SubstituteExpressions
)
from loki import ir, fgen
from loki.frontend.util import read_file

__all__ = [
    'MissingImplicitNoneRule', 'OnlyParameterGlobalVarRule', 'MissingIntfbRule',
    'MissingKindSpecifierRealLiterals'
]


class FindFloatLiterals(ExpressionFinder):
    """
    A visitor to collects :any:`FloatLiteral` used in an IR tree.

    See :class:`ExpressionFinder`
    """
    retriever = ExpressionRetriever(lambda e: isinstance(e, (FloatLiteral,)))

class MissingKindSpecifierRealLiterals(GenericRule):
    """
    ...
    """

    type = RuleType.SERIOUS
    fixable = True

    docs = {
        'id': 'L0',
        'title': (
            'Real Literals must have a kind specifier. '
        ),
    }


    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        ...
        """
        literal_nodes = FindFloatLiterals(with_ir_node=True).visit(subroutine.body)
        for node, literals in literal_nodes:
            for literal in literals:
                if literal.kind is None:
                    rule_report.add(f'Real/Float literal without kind specifier "{literal}"', node)

    @classmethod
    def fix_subroutinei_test_2(cls, subroutine, rule_report, config, sourcefile=None):
        """
        ...
        """
        for node, literals in literal_nodes:
            literal_map = {}
            for literal in literals:
                if literal.kind is None and 'e' not in literal.value.lower() and 'd' not in literal.value.lower():
                    literal_map[literal] = FloatLiteral(value=literal.value, kind='JPRB')
            if literal_map:
                # fixed_node = SubstituteExpressions(literal_map).visit(node)
                # for key in literal_map:
                #     fixed_node = re.sub(rf'{re.escape()}',
                #             , content, flags = re.S)
                indent = int((len(node.source.string) - len(node.source.string.lstrip(' ')))/2)
                fixed_node_str = fgen(fixed_node, depth=indent)
                with open (f'loki_lint_{subroutine.name}_new_file_fixed_node_str.F90', 'w') as f:
                    f.write(fixed_node_str)
                content_new = re.sub(rf'{re.escape(node.source.string)}',
                        fixed_node_str, content, flags = re.S)
                content = content_new
        with open (f'loki_lint_{subroutine.name}_new_file.F90', 'w') as f:
            f.write(content_new)
        diff = difflib.unified_diff(original_content.splitlines(), content_new.splitlines(),
                f'a/{sourcefile.path}', f'b/{sourcefile.path}', lineterm='')
        diff_str = '\n'.join(list(diff))
        # print(f"---{sourcefile.path}------")
        # print(diff_str)
        # print(f"--------------------------")
        with open (f'loki_lint_{subroutine.name}.approach_2.patch', 'w') as f:
            f.write(diff_str)
            f.write('\n')

    @classmethod
    def fix_subroutine_test(cls, subroutine, rule_report, config, sourcefile=None):
        """
        ...
        """
        # sourcefile = subroutine.source.file
        print(f"fix_subroutine: subroutine: {subroutine} | subroutine.source: {subroutine.source} | subroutine.source.file: {subroutine.source.file}")
        original_content = read_file(str(sourcefile.path))
        content = original_content
        literals = FindFloatLiterals(with_ir_node=False).visit(subroutine.body)
        literal_map = {}
        for literal in literals:
            if literal.kind is None:
                literal_map[literal] = FloatLiteral(value=literal.value, kind='JPRB')
        # content_new = content
        if literal_map:
            for key in literal_map:
                # print(f"replace ")
                # content_new = re.sub(rf'{re.escape(str(key))}', rf'{re.escape(str(literal_map[key]))}', content, flags = re.S)
                content_new = re.sub(rf'{re.escape(str(key))}', str(literal_map[key]), content, flags = re.S)
                content = content_new
        diff = difflib.unified_diff(original_content.splitlines(), content_new.splitlines(),
                f'a/{sourcefile.path}', f'b/{sourcefile.path}', lineterm='')
        diff_str = '\n'.join(list(diff))
        # print(f"---{sourcefile.path}------")
        # print(diff_str)
        # print(f"--------------------------")
        with open (f'loki_lint_{subroutine.name}.approach_2.patch', 'w') as f:
            f.write(diff_str)
            f.write('\n')
        
        """
        for node, literals in literal_nodes:
            literal_map = {}
            for literal in literals:
                if literal.kind is None:
                    literal_map[literal] = FloatLiteral(value=literal.value, kind='JPRB')
            if literal_map:
                # fixed_node = SubstituteExpressions(literal_map).visit(node)
                # indent = int((len(node.source.string) - len(node.source.string.lstrip(' ')))/2)
                # fixed_node_str = fgen(fixed_node, depth=indent)
                # with open (f'loki_lint_{subroutine.name}_new_file_fixed_node_str.F90', 'w') as f:
                #     f.write(fixed_node_str)
                # content_new = re.sub(rf'{re.escape(node.source.string)}',
                #         fixed_node_str, content, flags = re.S)
                # content = content_new
        with open (f'loki_lint_{subroutine.name}_new_file.F90', 'w') as f:
            f.write(content_new)
        diff = difflib.unified_diff(original_content.splitlines(), content_new.splitlines(),
                f'a/{sourcefile.path}', f'b/{sourcefile.path}', lineterm='')
        diff_str = '\n'.join(list(diff))
        # print(f"---{sourcefile.path}------")
        # print(diff_str)
        # print(f"--------------------------")
        with open (f'loki_lint_{subroutine.name}.approach_2.patch', 'w') as f:
            f.write(diff_str)
            f.write('\n')
        """

    @classmethod
    def fix_subroutine(cls, subroutine, rule_report, config, sourcefile=None):
        """
        ...
        """
        # sourcefile = subroutine.source.file
        print(f"fix_subroutine: subroutine: {subroutine} | subroutine.source: {subroutine.source} | subroutine.source.file: {subroutine.source.file}")
        original_content = read_file(str(sourcefile.path))
        content = original_content
        literal_nodes = FindFloatLiterals(with_ir_node=True).visit(subroutine.body)
        content_new = None
        for node, literals in literal_nodes:
            # print(f"node.source: {node.source.string} | {type(node.source.string)}")
            literal_map = {}
            for literal in literals:
                if literal.kind is None and 'e' not in str(literal.value).lower() and 'd' not in str(literal.value).lower():
                    literal_map[literal] = FloatLiteral(value=literal.value, kind='JPRB')
            if literal_map:
                # fixed_node = SubstituteExpressions(literal_map).visit(node)
                fixed_node = node.source.string
                # if hasattr(node, 'comment') and node.comment is not None:
                #     comment = node.comment
                #     fixed_node._update(comment=None)
                # else:
                #     comment = None 
                for key in literal_map:
                    fixed_node = re.sub(rf'{re.escape(str(key))}',
                        str(literal_map[key]), fixed_node, flags = re.S)
                # indent = int((len(node.source.string) - len(node.source.string.lstrip(' ')))/2)
                # fixed_node_str = fgen(fixed_node, depth=indent)
                # if comment is not None:
                #     fixed_node._update(comment=comment)
                fixed_node_str = str(fixed_node)
                # with open (f'loki_lint_{subroutine.name}_new_file_fixed_node_str.F90', 'w') as f:
                #     f.write(fixed_node_str)
                content_new = re.sub(rf'{re.escape(node.source.string)}',
                        fixed_node_str, content, flags = re.S)
                content = content_new
        # with open (f'loki_lint_{subroutine.name}_new_file.F90', 'w') as f:
        #     f.write(content_new)
        if content_new is not None:
            diff = difflib.unified_diff(original_content.splitlines(), content_new.splitlines(),
                    f'a/{sourcefile.path}', f'b/{sourcefile.path}', lineterm='')
            diff_str = '\n'.join(list(diff))
            # print(f"---{sourcefile.path}------")
            # print(diff_str)
            # print(f"--------------------------")
            with open (f'loki_lint_{subroutine.name}.patch', 'w') as f:
                f.write(diff_str)
                f.write('\n')

    @classmethod
    def fix_subroutine_working(cls, subroutine, rule_report, config, sourcefile=None):
        """
        ...
        """
        # sourcefile = subroutine.source.file
        print(f"fix_subroutine: subroutine: {subroutine} | subroutine.source: {subroutine.source} | subroutine.source.file: {subroutine.source.file}")
        original_content = read_file(str(sourcefile.path))
        content = original_content
        literal_nodes = FindFloatLiterals(with_ir_node=True).visit(subroutine.body)
        for node, literals in literal_nodes:
            # print(f"node.source: {node.source.string} | {type(node.source.string)}")
            literal_map = {}
            for literal in literals:
                if literal.kind is None:
                    literal_map[literal] = FloatLiteral(value=literal.value, kind='JPRB')
            if literal_map:
                fixed_node = SubstituteExpressions(literal_map).visit(node)
                indent = int((len(node.source.string) - len(node.source.string.lstrip(' ')))/2)
                fixed_node_str = fgen(fixed_node, depth=indent)
                with open (f'loki_lint_{subroutine.name}_new_file_fixed_node_str.F90', 'w') as f:
                    f.write(fixed_node_str)
                content_new = re.sub(rf'{re.escape(node.source.string)}',
                        fixed_node_str, content, flags = re.S)
                content = content_new
        with open (f'loki_lint_{subroutine.name}_new_file.F90', 'w') as f:
            f.write(content_new)
        diff = difflib.unified_diff(original_content.splitlines(), content_new.splitlines(),
                f'a/{sourcefile.path}', f'b/{sourcefile.path}', lineterm='')
        diff_str = '\n'.join(list(diff))
        # print(f"---{sourcefile.path}------")
        # print(diff_str)
        # print(f"--------------------------")
        with open (f'loki_lint_{subroutine.name}.patch', 'w') as f:
            f.write(diff_str)
            f.write('\n')

class MissingImplicitNoneRule(GenericRule):
    """
    ``IMPLICIT NONE`` must be present in all scoping units but may be omitted
    in module procedures.
    """

    type = RuleType.SERIOUS

    docs = {
        'id': 'L1',
        'title': (
            'IMPLICIT NONE must figure in all scoping units. '
            'Once per module is sufficient.'
        ),
    }

    _regex = re.compile(r'implicit\s+none\b', re.I)

    @classmethod
    def check_for_implicit_none(cls, ir_):
        """
        Check for intrinsic nodes that match the regex.
        """
        for intr in FindNodes(ir.Intrinsic).visit(ir_):
            if cls._regex.match(intr.text):
                break
        else:
            return False
        return True

    @classmethod
    def check_module(cls, module, rule_report, config):
        """
        Check for ``IMPLICIT NONE`` in the module's spec.
        """
        found_implicit_none = cls.check_for_implicit_none(module.spec)
        if not found_implicit_none:
            # No 'IMPLICIT NONE' intrinsic node was found
            rule_report.add('No `IMPLICIT NONE` found', module)

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        Check for ``IMPLICIT NONE`` in the subroutine's spec or an enclosing
        :any:`Module` scope.
        """
        found_implicit_none = cls.check_for_implicit_none(subroutine.ir)

        # Check if enclosing scopes contain implicit none
        scope = subroutine.parent
        while scope and not found_implicit_none:
            if isinstance(scope, Module) and hasattr(scope, 'spec') and scope.spec:
                found_implicit_none = cls.check_for_implicit_none(scope.spec)
            scope = scope.parent if hasattr(scope, 'parent') else None

        if not found_implicit_none:
            # No 'IMPLICIT NONE' intrinsic node was found
            rule_report.add('No `IMPLICIT NONE` found', subroutine)


class OnlyParameterGlobalVarRule(GenericRule):
    """
    Only parameters to be declared as global variables.
    """

    type = RuleType.SERIOUS

    docs = {
        'id': 'L3',
        'title': 'Only parameters to be declared as global variables.'
    }

    @classmethod
    def check_module(cls, module, rule_report, config):
        for decl in module.declarations:
            if not decl.symbols[0].type.parameter:
                msg = f'Global variable(s) declared that are not parameters: {", ".join(s.name for s in decl.symbols)}'
                rule_report.add(msg, decl)


class MissingIntfbRule(GenericRule):
    """
    Calls to subroutines and functions that are provided neither by a module
    nor by a CONTAINS statement, must have a matching explicit interface block.
    """

    type = RuleType.SERIOUS

    docs = {
        'id': 'L9',
        'title': (
            'Explicit interface blocks required for procedures that are not '
            'imported or internal subprograms'
        )
    }

    @classmethod
    def _get_external_symbols(cls, program_unit):
        """
        Collect all imported symbols in :data:`program_unit` and
        parent scopes and return as a set of lower-case names
        """
        external_symbols = {name.lower() for name in _intrinsic_fortran_names}

        if program_unit.parent:
            external_symbols |= cls._get_external_symbols(program_unit.parent)

        # Get imported symbols
        external_symbols |= {
            s.name.lower()
            for import_ in program_unit.imports
            for s in import_.symbols or ()
        }

        # Collect all symbols declared via intfb includees
        c_includes = [
            include for include in FindNodes(ir.Import).visit(program_unit.ir)
            if include.c_import
        ]
        external_symbols |= {
            include.module[:-8].lower() for include in c_includes
            if include.module.endswith('.intfb.h')
        }
        external_symbols |= {
            include.module[:-7].lower() for include in c_includes
            if include.module.endswith('.func.h')
        }

        # Add locally declared interface symbols
        external_symbols |= {s.name.lower() for s in program_unit.interface_symbols}

        # Add internal subprograms and module procedures
        for routine in program_unit.routines:
            external_symbols.add(routine.name.lower())

        return external_symbols

    @staticmethod
    def _add_report(rule_report, node, call_name):
        """
        Register a missing interface block for a call to :data:`call_name`
        in the :any:`RuleReport`
        """
        msg = f'Missing import or interface block for called procedure `{call_name}`'
        rule_report.add(msg, node)

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        Check all :any:`CallStatement` and :any:`InlineCall` for a matching import
        or interface block.
        """
        external_symbols = cls._get_external_symbols(subroutine)

        # Collect all calls to routines without a corresponding symbol
        missing_calls = defaultdict(list)

        for call in FindNodes(ir.CallStatement).visit(subroutine.body):
            if not call.name.parent and str(call.name).lower() not in external_symbols:
                missing_calls[str(call.name).lower()] += [call]

        for node, calls in FindInlineCalls(with_ir_node=True).visit(subroutine.body):
            for call in calls:
                if not call.function.parent and call.name.lower() not in external_symbols:
                    missing_calls[call.name.lower()] += [node]

        # Create reports for each missing routine only for the first occurence
        for name, calls in missing_calls.items():
            cls._add_report(rule_report, calls[0], name)
