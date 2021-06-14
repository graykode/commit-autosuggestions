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

import os
import re
import json
import argparse
import jsonlines
from tqdm import tqdm
from functools import partial
from collections import Counter
from multiprocessing import Pool
from pydriller import RepositoryMining
from tree_sitter import Language, Parser
from typing import List, Dict, Any, Set, Optional

language_ext = {
    'python' : ['.py'],
    'javascript' : ['.js'],
    'go' : ['.go'],
    'java' : ['.java'],
    'ruby' : ['.rb'],
    'php' : ['.php']
}

DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")

def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr) if t is not None and len(t) > 0]

def tokenize_code(PARSER, blob: str) -> List:
    tokens = []
    tree = PARSER.parse(blob.encode())
    for child in tree.root_node.children:
        tokens += [t for t in tokenize_ast(child, blob) if t != ""]
    return tokens

def tokenize_ast(node, blob: str) -> List:
    tokens = []
    traverse(node, tokens)
    return [match_from_span(token, blob) for token in tokens]

def traverse(node, results: List) -> None:
    if node.type == 'string':
        results.append(node)
        return
    for n in node.children:
        traverse(n, results)
    if not node.children:
        results.append(node)

def match_from_span(node, blob: str) -> str:
    lines = blob.split('\n')
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
    else:
        return lines[line_start][char_start:char_end]

def message_cleaner(message):
    msg = message.split("\n")[0]
    msg = re.sub(r"(\(|\[|\{|\<)#([0-9])+(\)|\]|\}|\>)", "", msg)
    return msg

def get_code_diff(commit, PARSER, args):

    n_files = 0
    addeds, deleteds = [], []
    for mod in commit.modifications:
        if len(addeds) == args.max_duplicate + 1 and len(deleteds) == args.max_duplicate + 1:
            return (addeds, deleteds, n_files) # to more fast

        # skip only if add or delete file
        if not (mod.old_path and mod.new_path):
            continue
        if os.path.splitext(mod.new_path)[1] not in language_ext[args.lang]:
            continue
        if not mod.diff_parsed["added"]:
            continue
        if not mod.diff_parsed["deleted"]:
            continue
        if len("".join([code for line, code in mod.diff_parsed["added"]]).encode("utf8")) > 1 * 1024 * 1024:
            continue
        if len("".join([code for line, code in mod.diff_parsed["deleted"]]).encode("utf8")) > 1 * 1024 * 1024:
            continue

        try:
            added, deleted = [], []
            for line, code in mod.diff_parsed["added"]:
                added += ['<s>'] + tokenize_code(PARSER, code)
            for line, code in mod.diff_parsed["deleted"]:
                deleted += ['<s>'] + tokenize_code(PARSER, code)
        except:
            continue

        if added and deleted and len(added) + len(deleted) <= args.max_source_length:
            addeds.append(added)
            deleteds.append(deleted)
            n_files += 1

    assert len(addeds) == len(deleteds) == n_files
    return (addeds, deleteds, n_files)

def jobs(repo_path, args):
    PARSER = Parser()
    PARSER.set_language(Language(args.tree_sitter, args.lang))

    n_file_per_commit = Counter()
    add_tokens_per_del_tokens = []

    if os.path.exists(repo_path):
        submodule = os.path.join(repo_path, '.gitmodules')
        if os.path.exists(submodule):
            os.remove(submodule)

        try:
            n_stored_commit = 0
            for commit in RepositoryMining(
                repo_path,
                only_no_merge=True,
                only_in_branch='master',
                only_modifications_with_file_types=language_ext[args.lang]
            ).traverse_commits():
                if n_stored_commit > args.max_commit_number:
                    break

                cleaned_message = message_cleaner(commit.msg)
                if not cleaned_message:
                    continue
                commit_tokens = tokenize_docstring_from_string(cleaned_message)

                if len(commit_tokens) < args.min_target_length:
                    continue

                addeds, deleteds, n_files = get_code_diff(commit, PARSER, args)
                if 1 <= n_files and n_files <= args.max_duplicate:
                    with jsonlines.open(args.output_file, mode="a") as writer:
                        writer.write(
                            {
                                "commit_tokens": commit_tokens,
                                "add_tokens": addeds[0],
                                "del_tokens": deleteds[0],
                            }
                        )
                    add_tokens_per_del_tokens.append( len(addeds[0]) / len(deleteds[0]) )
                    n_file_per_commit.update({n_files})
                    n_stored_commit += 1
        except:
            pass

    return (n_file_per_commit, add_tokens_per_del_tokens)

def read_all_repos(repos_dir, lang):
    repos = set()
    download_folder = os.path.join(repos_dir, lang)

    for owner in os.listdir(download_folder):
        for repo in os.listdir(os.path.join(download_folder, owner)):
            repos.add(os.path.join(download_folder,owner,repo))

    return list(repos)

def main(args):
    n_file_per_commit = Counter()
    add_tokens_per_del_tokens = []

    repos = read_all_repos(repos_dir=args.repos_dir, lang=args.lang)
    print(f"{args.lang} : {len(repos)} !")

    args.output_dir = os.path.join(args.output_dir, args.lang)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, 'dataset.jsonl')

    # jobs(repos[0], args)
    pool = Pool(args.num_workers)
    func = partial(jobs, args=args)
    with tqdm(total=len(repos)) as pbar:
        for i, (x, y) in tqdm(enumerate(pool.imap_unordered(func, repos))):
            pbar.update()
            n_file_per_commit.update(x)
            add_tokens_per_del_tokens += y
    pool.close()
    pool.join()

    with open(os.path.join(args.output_dir, f'{args.lang}.json'), "w") as json_file:
        json.dump({
            'n_repos': len(repos),
            'n_file_per_commit': dict(sorted(n_file_per_commit.items(), key=lambda item: item[0])),
            'add_tokens_per_del_tokens': sorted(add_tokens_per_del_tokens)
        }, json_file)

if __name__ == "__main__":
    ## Required parameters
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--repos_dir", type=str, required=True,
                        help="directory that all repositories had been downloaded.",)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the preprocessed data will be written.")
    parser.add_argument("--tree_sitter", type=str, required=True,
                        help="The path of py-tree-sitter-languages.so (ex, /src/build/py-tree-sitter-languages.so)")
    parser.add_argument("--lang", type=str, required=True,
                        choices=['python', 'javascript', 'go', 'java', 'ruby', 'php'],
                        help="programming language")

    parser.add_argument("--num_workers", default=4, type=int, help="number of process")
    parser.add_argument("--min_target_length", default=5, type=int,
                        help="The minimum of target's length")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum of source's length")
    parser.add_argument("--max_duplicate", default=2, type=int,
                        help="The maximum of number of file per commit")
    parser.add_argument("--max_commit_number", default=100, type=int,
                        help="The minimum of number of commit in an one repository")
    args = parser.parse_args()
    print(args)

    main(args)
