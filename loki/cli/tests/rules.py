# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.lint import GenericRule, RuleType


class CliDummyRule(GenericRule):
    """
    Dummy rule for testing the loki-lint command line interface.
    """

    type = RuleType.WARN

    docs = {
        'id': 'CLI.1',
        'title': 'A dummy rule for testing the loki-lint CLI'
    }

    config = {'dummy_key': 'dummy value'}

    @classmethod
    def check_file(cls, sourcefile, rule_report, config):
        """
        Report source files whose name is marked as violating.
        """
        if sourcefile.path.name.startswith('violating'):
            rule_report.add('Dummy file violation', sourcefile)
