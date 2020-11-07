# Copyright 2020-present Tae Hwan Jung
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import json
import requests
import subprocess
import configparser
import whatthepatch
from os.path import expanduser, join, exists

def get_diff_from_project():
    proc = subprocess.Popen(["git", "diff", "--cached"], stdout=subprocess.PIPE)
    staged_files = proc.stdout.readlines()
    staged_files = [f.decode("utf-8") for f in staged_files]
    assert staged_files, "You have to update the file via `git add` to change not staged for commit."
    return staged_files

def commit_message_parser(messages):
    result = []
    for idx, (path, commit) in enumerate(messages.items()):
        click.echo("  - " + " ".join(commit["message"]))
        result.append(" ".join(commit["message"]))
    return result

def healthcheck(endpoint):
    response = requests.get(
        f"{endpoint}/",
        headers={'Content-Type': 'application/json; charset=utf-8'}
    )
    assert response.status_code == 200, f"{endpoint} is not running."

def tokenizing(code, endpoint):
    data = {"code": code }
    res = requests.post(
        f'{endpoint}/tokenizer',
        data=json.dumps(data),
        headers={'Content-Type': 'application/json; charset=utf-8'}
    )
    return json.loads(res.text)["tokens"]

def commit_autosuggestions(diffs, endpoint):
    commit_message = {}
    for idx, example in enumerate(whatthepatch.parse_patch(diffs)):
        if not example.changes:
            continue

        isadded, isdeleted = False, False
        added, deleted = [], []
        for change in example.changes:
            if change.old == None and change.new != None:
                added.append(change.line)
                isadded = True
            elif change.old != None and change.new == None:
                deleted.append(change.line)
                isdeleted = True

        # To speed up tokenizing request.
        added = tokenizing(" ".join(added), endpoint=endpoint)
        deleted = tokenizing(" ".join(deleted), endpoint=endpoint)

        _path = example.header.new_path \
            if example.header.new_path \
            else example.header.old_path
        if _path:
            if isadded and not isdeleted:
                data = {"idx": idx, "added" : added, "deleted" : deleted}
                res = requests.post(
                    f'{endpoint}/added',
                    data=json.dumps(data),
                    headers={'Content-Type': 'application/json; charset=utf-8'}
                )
                commit = json.loads(res.text)
                commit_message[_path] = commit
            else:
                data = {"idx": idx, "added": added, "deleted": deleted}
                res = requests.post(
                    f'{endpoint}/diff',
                    data=json.dumps(data),
                    headers={'Content-Type': 'application/json; charset=utf-8'}
                )
                commit = json.loads(res.text)
                commit_message[_path] = commit
    return commit_message

def commit(messages):
    m = []
    for msg in messages:
        m.extend(["-m", msg])
    subprocess.Popen(["git", "commit"] + m, stdout=subprocess.PIPE)

@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--profile', default='default', type=str,
    help='unique name for managing each independent settings')
@click.option('--file', '-f', type=click.File('r'),
    help='patch file containing git diff '
         '(e.g. file created by `git add` and `git diff --cached > test.diff`)')
@click.option('--verbose', '-v', is_flag=True,
    help='print suggested commit message more detail.')
@click.option('--autocommit', '-a', is_flag=True,
    help='automatically commit without asking if you want to commit')
def cli(ctx, profile, file, verbose, autocommit):
    if not ctx.invoked_subcommand:
        profile = profile.upper()
        path = join(expanduser("~"), '.commit-autosuggestions.ini')
        config = configparser.ConfigParser()
        if not exists(path):
            raise FileNotFoundError("The configuration file for commit-autosuggestions could not be found. "
                                    "Enter the `commit configure --help` command.")
        config.read(path)
        if profile.upper() not in list(config.keys()):
            raise KeyError(f"That profile({profile}) cannot be found in the configuration file. Check the {path}.")

        endpoint = config[profile]['endpoint']
        healthcheck(endpoint=endpoint)

        staged_files = file if file else get_diff_from_project()
        staged_files = [f.strip() for f in staged_files]
        diffs = "\n".join(staged_files)

        result = commit_autosuggestions(diffs=diffs, endpoint=endpoint)
        if verbose:
            click.echo(
                json.dumps(result, indent=4, sort_keys=True) + "\n"
            )

        click.echo(click.style('[INFO]', fg='green') + " The generated message is as follows:")
        messages = commit_message_parser(result)

        if autocommit or click.confirm('Do you want to commit this message?'):
            commit(messages)


@cli.command()
@click.option('--profile', default='default', type=str,
    help='unique name for managing each independent settings')
@click.option('--endpoint', required=True, type=str,
    help='endpoint address accessible to the server (example : http://127.0.0.1:5000/)')
def configure(profile, endpoint):
    profile = profile.upper()
    path = join(expanduser("~"), '.commit-autosuggestions.ini')
    config = configparser.ConfigParser()
    if exists(path):
        config.read(path)
    if profile not in list(config.keys()):
        config[profile] = {}
    if endpoint:
        config[profile]['endpoint'] = endpoint

    click.echo(f"configure of commit-autosuggestions is setted up in {path}.")
    with open(path, 'w') as configfile:
        config.write(configfile)
    for key in config[profile]:
        click.echo("[" + click.style('username', fg='blue') + "] : " + profile)
        click.echo("[" + click.style(key, fg='green') + "] : " + config[profile][key])
        click.echo()

if __name__ == '__main__':
    cli()