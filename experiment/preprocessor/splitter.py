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
import spacy
import random
import argparse
import jsonlines
from tqdm import tqdm
from transformers import RobertaTokenizer

def write_jsonl(lines, path, mode):
    saved_path = os.path.join(path, mode)
    for line in lines:
        with jsonlines.open(f"{saved_path}.jsonl", mode="a") as writer:
            writer.write(line)
    print(f'write to {saved_path}.jsonl!')

def main(args):
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    args.output_dir = os.path.join(args.output_dir, args.lang)
    args.output_file = os.path.join(args.output_dir, 'dataset.jsonl')
    data = []
    num_lines = sum(1 for line in open(args.output_file, 'r'))
    with open(args.output_file, encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), total=num_lines):
            line = line.strip()
            line = json.loads(line)

            added = ' '.join(line['add_tokens']).replace('\n', ' ')
            added = ' '.join([v for v in added.strip().split() if v != '<s>'])

            deleted = ' '.join(line['del_tokens']).replace('\n', ' ')
            deleted = ' '.join([v for v in deleted.strip().split() if v != '<s>'])

            if len(tokenizer.tokenize(added)) > 512:
                continue
            if len(tokenizer.tokenize(deleted)) > 512:
                continue

            msg = " ".join(line["commit_tokens"])
            msg = re.sub(r"#([0-9])+", "", msg)
            pos = [p.pos_ for p in nlp(msg)]
            if pos[0] == "VERB":
                commit_type = [p.lemma_ for p in nlp(msg)][0].lower()

                line['add_tokens'] = added
                line['del_tokens'] = deleted
                line['commit_type'] = commit_type

                data.append(line)

    random.shuffle(data)
    n_data = len(data)
    write_jsonl(
        data[:int(n_data * 0.8)],
        path=args.output_dir, mode='train'
    )
    write_jsonl(
        data[int(n_data * 0.8):int(n_data * 0.9)],
        path=args.output_dir, mode='valid'
    )
    write_jsonl(
        data[int(n_data * 0.9):],
        path=args.output_dir, mode='test'
    )

    print(f'split is done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the preprocessed data will be written.")
    parser.add_argument("--lang", type=str, required=True,
                        choices=['python', 'javascript', 'go', 'java', 'ruby', 'php'],
                        help="programming language")
    args = parser.parse_args()
    print(args)

    nlp = spacy.load("en_core_web_sm")

    main(args)