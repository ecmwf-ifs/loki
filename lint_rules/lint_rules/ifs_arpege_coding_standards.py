# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Implementation of rules from the IFS Arpege coding standards as :any:`GenericRule`

See
"""


from loki import (
    FindInlineCalls, FindNodes, GenericRule, RuleType
)
import loki.ir as ir


__all__ = [
    'MissingIntfbRule',
]


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
        external_symbols = set()

        if program_unit.parent:
            external_symbols |= cls._get_external_symbols(program_unit.parent)

        # Get imported symbols
        external_symbols |= {
            s.name.lower()
            for import_ in program_unit.imports
            for s in import_.symbols or ()
        }

        # Collect all symbols declared via intfb includees
        external_symbols |= {
            include.module[:-8].lower()
            for include in FindNodes(ir.Import).visit(program_unit.ir)
            if include.c_import and include.module.endswith('intfb.h')
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
        msg = f'Missing import or interface block for called procedure {call_name}'
        rule_report.add(msg, node)

    @classmethod
    def check_subroutine(cls, subroutine, rule_report, config, **kwargs):
        """
        Check all :any:`CallStatement` and :any:`InlineCall` for a matching import
        or interface block.
        """
        external_symbols = cls._get_external_symbols(subroutine)

        for call in FindNodes(ir.CallStatement).visit(subroutine.body):
            if str(call.name).lower() not in external_symbols:
                cls._add_report(rule_report, call, call.name)

        for node, calls in FindInlineCalls(with_ir_node=True).visit(subroutine.body):
            for call in calls:
                if call.name.lower() not in external_symbols:
                    cls._add_report(rule_report, node, call.name)
