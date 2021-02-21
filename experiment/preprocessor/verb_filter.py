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
from glob import glob
from tqdm import tqdm
from collections import Counter

verbs = ['add', 'create', 'make', 'implement', 'fix', 'remove', 'update', 'upgrade', 'use', 'move', 'change', 'improve', 'allow']

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def main(args):
    # create verb count
    data = {}
    c = Counter()
    for file in glob(f'{args.input_dir}/*/*.jsonl'):
        lines = []
        num_lines = sum(1 for line in open(file, 'r'))
        with open(file, encoding="utf-8") as f:
            for idx, line in tqdm(enumerate(f), total=num_lines):
                line = line.strip()
                line = json.loads(line)

                p1 = re.compile('#([0-9])+')
                if p1.search("".join(line["commit_tokens"])) is not None:
                    continue

                p2 = re.compile('([0-9])+\.([0-9])+')
                if p2.search("".join(line["commit_tokens"])) is not None:
                    continue

                # find long word such as hashid
                if sum(1 for _ in filter(lambda x: x >= 32, [len(_a) for _a in line["commit_tokens"]])):
                    continue

                msg = " ".join(line["commit_tokens"])
                if not is_ascii(msg):
                    continue

                lines.append(line)
                c.update({line['commit_type']})
        data[file] = lines

    topk = 200
    s = 0
    t = sum([v for k, v in c.items()])
    for i, (k, v) in enumerate(sorted(c.items(), key=lambda item: item[1], reverse=True)[:topk]):
        s += v
        print(i + 1, k, v / t * 100, s / t * 100)
    print(i + 2, 'unk', 100 - s / t * 100)

    print('\nReady for Write files!!')
    for file, lines in data.items():
        _, lang, filename = file.split('/')
        os.makedirs(os.path.join(args.output_dir, lang), exist_ok=True)
        saved_path = os.path.join(args.output_dir, lang, filename)
        for line in lines:
            if line['commit_type'] not in verbs:
                continue
            with jsonlines.open(f"{saved_path}", mode="a") as writer:
                writer.write(line)
        print(f'write to {saved_path}!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="The input directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the preprocessed data will be written.")
    args = parser.parse_args()
    print(args)

    main(args)