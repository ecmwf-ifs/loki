#!/usr/bin/env python3

import os
import click
from sites.toolkit.file_manager import Authenticator, FileManager, Site

class NotRequiredIf(click.Option):
    """
    Custom option class that makes an option not required if some condition
    is fulfilled.

    Source: https://stackoverflow.com/a/44349292
    """
    def __init__(self, *args, **kwargs):
        self.not_required_if = kwargs.pop('not_required_if')
        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs['help'] = (kwargs.get('help', '') +
            ' NOTE: This argument is mutually exclusive with %s' % self.not_required_if
        ).strip()
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        we_are_present = self.name in opts
        other_present = self.not_required_if in opts

        if other_present:
            if we_are_present:
                raise click.UsageError(
                    "Illegal usage: `%s` is mutually exclusive with `%s`" % (
                        self.name, self.not_required_if))
            self.prompt = None

        return super().handle_parse_result(ctx, opts, args)


def get_file_manager(ctx):
    """Connect to the Sites hub and instantiate a file manager for chosen space and site."""
    if not ctx.obj['space']:
        raise click.UsageError('No space given. Provide "--space" before the command.')
    if not ctx.obj['name']:
        raise click.UsageError('No site name given. Provide "--name" before the command.')

    # Authenticate against Site
    if ctx.obj['token']:
        my_authenticator = Authenticator.from_token(token=ctx.obj['token'])
    else:
        if not ctx.obj['password']:
            ctx.obj['password'] = click.prompt('Password', hide_input=True)
        my_authenticator = Authenticator.from_credentials(username=ctx.obj['username'],
                                                          password=ctx.obj['password'])

    # Connect to the selected site...
    if ctx.obj['use_stage']:
        my_site = Site(space=ctx.obj['space'], name=ctx.obj['name'],
                       authenticator=my_authenticator,
                       sites_base_url='https://sites-stage.ecmwf.int')
    else:
        my_site = Site(space=ctx.obj['space'], name=ctx.obj['name'],
                       authenticator=my_authenticator)

    # ...and create a file manager
    my_site_manager = FileManager(site=my_site)

    return my_site_manager


@click.group()
@click.option('--space', help='Name of the space in the Sites hub')
@click.option('--name', help='Name of the site in the space')
@click.option('--token', help='API authentication token')
@click.option('--username', help='Username for authentication',
              default=lambda: os.environ.get('USER', ''), show_default='current user',
              cls=NotRequiredIf, not_required_if='token')
@click.option('--password', help='Password for authentication')#, prompt=True,
#              hide_input=True, cls=NotRequiredIf, not_required_if='token')
@click.option('--use-stage', help='Use staging hub (https://sites-stage.ecmwf.int)',
              is_flag=True, default=False)
@click.pass_context
def cli(ctx, space, name, token, username, password, use_stage):
    """Interact with the Sites hub. """
    ctx.obj['space'] = space
    ctx.obj['name'] = name
    ctx.obj['token'] = token
    ctx.obj['username'] = username
    ctx.obj['password'] = password
    ctx.obj['use_stage'] = use_stage


@cli.command(short_help='Upload a local path')
@click.argument('local_path')
@click.argument('remote-path', required=False, default='')
@click.option('--clean/--no-clean', help='Clean upload path first', default=False)
@click.pass_context
def upload(ctx, local_path, remote_path, clean):
    """Upload LOCAL_PATH to REMOTE_PATH.

    LOCAL_PATH is a local file or directory that is uploaded into REMOTE_PATH
    in the sites hub. If no REMOTE_PATH is given, LOCAL_PATH is uploaded into
    the root folder of the site.
    """
    my_site_manager = get_file_manager(ctx)
    if clean:
        my_site_manager.delete(remote_path=remote_path, recursive=True)
    my_site_manager.upload(local_path=local_path, remote_path=remote_path)


@cli.command(short_help='Download a remote path')
@click.argument('remote-path')
@click.pass_context
def download(ctx, remote_path):
    """Download REMOTE_PATH."""
    my_site_manager = get_file_manager(ctx)
    my_site_manager.download(remote_path=remote_path, recursive=True)


@cli.command(short_help='Delete a remote path')
@click.argument('remote-path')
@click.pass_context
def delete(ctx, remote_path):
    """Delete REMOTE_PATH."""
    my_site_manager = get_file_manager(ctx)
    my_site_manager.delete(remote_path=remote_path, recursive=True)


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
