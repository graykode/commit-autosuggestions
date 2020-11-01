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
import random
import jsonlines
import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import Pool
from transformers import RobertaTokenizer
from pydriller import RepositoryMining

def message_cleaner(message):
    msg = message.split("\n")[0]
    msg = re.sub(r"(\(|)#([0-9])+(\)|)", "", msg)
    return msg


def jobs(repo, args):
    repo_path = os.path.join(args.repos_dir, repo)
    if os.path.exists(repo_path):
        for commit in RepositoryMining(
            repo_path, only_modifications_with_file_types=['.py']
        ).traverse_commits():
            cleaned_message = message_cleaner(commit.msg)
            tokenized_message = args.tokenizer.tokenize(cleaned_message)
            if len(tokenized_message) > args.max_target_length:
                continue

            for mod in commit.modifications:
                if not (mod.old_path and mod.new_path):
                    continue
                if os.path.splitext(mod.new_path)[1] != '.py':
                    continue
                if not mod.diff_parsed["added"]:
                    continue
                if not mod.diff_parsed["deleted"]:
                    continue

                added, deleted = [], []

                for line, code in mod.diff_parsed["added"]:
                    added.extend(args.tokenizer.tokenize(code))

                for line, code in mod.diff_parsed["deleted"]:
                    deleted.extend(args.tokenizer.tokenize(code))

                if added and deleted and len(added) + len(deleted) <= args.max_source_length - 3:
                    with jsonlines.open(args.output_file, mode="a") as writer:
                        writer.write(
                            {
                                "msg": tokenized_message,
                                "added": added,
                                "deleted": deleted,
                            }
                        )

def write_jsonl(lines, path, mode):
    saved_path = os.path.join(path, mode)
    for line in lines:
        with jsonlines.open(f"{saved_path}.jsonl", mode="a") as writer:
            writer.write(line)

def main(args):
    repos = set()
    with open(args.repositories, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            repos.add(line.replace('https://github.com/', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, 'dataset.jsonl')

    func = partial(jobs, args=args)
    with Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(repos)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(func, repos))):
                pbar.update()

    data = []
    with open(args.output_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            data.append(json.loads(line))

    random.shuffle(data)
    n_data = len(data)
    write_jsonl(
        data[:int(n_data * 0.9)],
        path=args.output_dir, mode='train'
    )
    write_jsonl(
        data[int(n_data * 0.9):int(n_data * 0.95)],
        path=args.output_dir, mode='valid'
    )
    write_jsonl(
        data[int(n_data * 0.95):],
        path=args.output_dir, mode='test'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--repositories", type=str, required=True,
                        help="repositories file path.")
    parser.add_argument("--repos_dir", type=str, required=True,
                        help="directory that all repositories had been downloaded.",)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the preprocessed data will be written.")
    parser.add_argument("--tokenizer_name", type=str,
                        default="microsoft/codebert-base", help="The name of tokenizer",)
    parser.add_argument("--num_workers", default=4, type=int, help="number of process")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    args = parser.parse_args()

    args.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    main(args)
