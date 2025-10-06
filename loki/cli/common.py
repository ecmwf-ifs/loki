# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path
from typing import Tuple

import click
from click_option_group import optgroup

from loki.frontend import Frontend
from loki.tools.util import auto_post_mortem_debugger, set_excepthook


__all__ = ['cli', 'frontend_options']


@click.group()
@click.option(
    '--debug/--no-debug', default=False, show_default=True,
    help=('Enable / disable debug mode. This automatically attaches '
          'a debugger when exceptions occur')
)
def cli(debug):
    if debug:
        set_excepthook(hook=auto_post_mortem_debugger)


@dataclass
class FrontendOptions:
    """
    Storage object for frontend options that can be passed to the :any:`Scheduler`.
    """

    frontend: Frontend = Frontend.FP
    preprocess: bool = False
    includes: Tuple[Path] = ()
    defines: Tuple[str] = ()
    xmods: Tuple[Path] = ()
    omni_includes: Tuple[str] = ()

    @property
    def asdict(self):
        return asdict(self)


def frontend_options(func):
    """
    Option group configuring the Loki frontend options, including preprocessing.
    """

    @optgroup.group('Loki frontend options',
                    help='Frontend parsing options for Loki.')
    @optgroup.option('--frontend', default='fp', type=click.Choice(['fp', 'ofp', 'omni']),
                     help='Frontend parser to use (default FP)')
    @optgroup.option('--cpp/--no-cpp', default=False,
                     help='Trigger C-preprocessing of source files.')
    @optgroup.option('--include', '-I', type=click.Path(), multiple=True,
                     help='Path for additional header file(s)')
    @optgroup.option('--define', '-D', multiple=True,
                     help='Additional symbol definitions for the C-preprocessor')
    @optgroup.option('--xmod', '-M', type=click.Path(), multiple=True,
                     help='Path for additional .xmod file(s) for OMNI')
    @optgroup.option('--omni-include', type=click.Path(), multiple=True,
                     help='Additional path for header files, specifically for OMNI')
    @click.pass_context
    @wraps(func)
    def process_frontend_options(ctx, *args, **kwargs):
        frontendopts = ctx.ensure_object(FrontendOptions)
        frontendopts.frontend = Frontend[kwargs.pop('frontend').upper()]
        frontendopts.preprocess = kwargs.pop('cpp')
        frontendopts.includes = kwargs.pop('include')
        frontendopts.defines = kwargs.pop('define')
        frontendopts.xmods = kwargs.pop('xmod')
        frontendopts.omni_includes = kwargs.pop('omni_include')
        return ctx.invoke(func, *args, frontendopts, **kwargs)

    return process_frontend_options
