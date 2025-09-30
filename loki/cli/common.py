# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import click

from loki.tools.util import auto_post_mortem_debugger, set_excepthook


__all__ = ['cli']


@click.group()
@click.option(
    '--debug/--no-debug', default=False, show_default=True,
    help=('Enable / disable debug mode. This automatically attaches '
          'a debugger when exceptions occur')
)
def cli(debug):
    if debug:
        set_excepthook(hook=auto_post_mortem_debugger)
