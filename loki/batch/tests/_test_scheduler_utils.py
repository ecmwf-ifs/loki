# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
import re


class VisGraphWrapper:
    """
    Testing utility to parse the generated callgraph visualisation.
    """

    _re_nodes = re.compile(r'\s*\"?(?P<node>[\w%#./-]+)\"? \[colo', re.IGNORECASE)
    _re_edges = re.compile(r'\s*\"?(?P<parent>[\w%#./-]+)\"? -> \"?(?P<child>[\w%#./-]+)\"?', re.IGNORECASE)

    def __init__(self, path):
        with Path(path).open('r') as f:
            self.text = f.read()

    @property
    def nodes(self):
        return list(self._re_nodes.findall(self.text))

    @property
    def edges(self):
        return list(self._re_edges.findall(self.text))
