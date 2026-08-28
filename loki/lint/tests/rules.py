# SPDX-FileCopyrightText: 2018 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileComment: In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki.lint import GenericRule, RuleType

__all__ = ['DummyRule']


class DummyRule(GenericRule):

    type = RuleType.WARN

    docs = {'title': 'A dummy rule for the sake of testing the Linter'}

    config = {'dummy_key': 'dummy value'}
