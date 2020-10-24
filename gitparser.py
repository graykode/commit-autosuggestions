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
import json
import jsonlines
import argparse
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from multiprocessing.pool import Pool
from transformers import RobertaTokenizer
from pydriller import GitRepository, RepositoryMining


def jobs(repo_paths, args):
    repo, paths = repo_paths
    repo_path = os.path.join(args.repos_dir, repo)

    if os.path.exists(repo_path):
        gr = GitRepository(repo_path)

        for path in paths:
            commits = gr.get_commits_modified_file(path)
            for commit in RepositoryMining(
                repo_path, only_commits=commits
            ).traverse_commits():
                message = (commit.msg).split("\n")[0]

                added, deleted = [], []
                for mod in commit.modifications:
                    if mod.new_path == path:
                        for line, code in mod.diff_parsed["added"]:
                            added += args.tokenizer.tokenize(code)
                            assert isinstance(added, list)

                        for line, code in mod.diff_parsed["deleted"]:
                            deleted += args.tokenizer.tokenize(code)
                            assert isinstance(deleted, list)

                        with jsonlines.open(args.output_file, mode="a") as writer:
                            writer.write(
                                {
                                    "repo": repo,
                                    "path": path,
                                    "sha": commit.hash,
                                    "msg": args.tokenizer.tokenize(message),
                                    "added": added,
                                    "deleted": deleted,
                                }
                            )

def main(args):
    repos = defaultdict(list)
    with open(args.jsonl_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            repos[js["repo"]].append(js["path"])

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, os.path.basename(args.jsonl_file))

    func = partial(jobs, args=args)
    with Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(repos)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(func, repos.items()))):
                pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--jsonl_file", type=str, required=True, help="jsonl file path."
    )
    parser.add_argument(
        "--repos_dir",
        type=str,
        required=True,
        help="directory that all repositories will be downloaded.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the preprocessed data will be written.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="microsoft/codebert-base",
        help="The name of tokenizer",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of process",
    )

    args = parser.parse_args()

    args.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    main(args)
