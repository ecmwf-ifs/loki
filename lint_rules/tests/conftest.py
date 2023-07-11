# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from loki import HAVE_FP, FP
from loki.lint import Reporter, Linter


__all__ = ['available_frontends', 'run_linter']


def available_frontends():
    """Choose frontend to use (Linter currently relies exclusively on Fparser)"""
    if HAVE_FP:
        return [FP,]
    return []


def run_linter(sourcefile, rule_list, config=None, handlers=None, targets=None):
    """
    Run the linter for the given source file with the specified list of rules.
    """
    reporter = Reporter(handlers)
    linter = Linter(reporter, rules=rule_list, config=config)
    linter.check(sourcefile, targets=targets)
    return linter
