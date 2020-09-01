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
import enum
import logging
import argparse
import whatthepatch
from git import Repo
from functools import partial
from multiprocessing.pool import Pool
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class PATCH(enum.Enum):
	PLUS=1
	MINUS=2

def truncate(tuple, max_length, value=0):
    ls = []
    for t in tuple:
        if isinstance(t, int):
            t = [t]
        ls.extend(t)
    ls = ls[:max_length - 1]
    ls.insert(0, value)
    if len(ls) < max_length:
        ls.extend([0] * (max_length - len(ls)))
    assert len(ls) == max_length
    return ls

def encode_line(tokenizer, line, patch):
    tokens = tokenizer.tokenize(line)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return (
        tokens,
        [1] * len(tokens),
        len(tokens) * [patch.value]
    )

def sha_parse(sha, tokenizer, max_length=1024):

    chunks = []
    for diff in whatthepatch.parse_patch(repo.git.show(sha)):
        if diff.header.old_path != diff.header.new_path:
            chunks.append(encode_line(tokenizer, diff.header.old_path, PATCH.MINUS))
            chunks.append(encode_line(tokenizer, diff.header.new_path, PATCH.PLUS))
        if not diff.changes:
            continue
        for change in diff.changes:
            if change.old == None and change.new != None:
                chunks.append(encode_line(tokenizer, change.line, PATCH.PLUS))
            elif change.old != None and change.new == None:
                chunks.append(encode_line(tokenizer, change.line, PATCH.PLUS))

    input_ids, attention_masks, patch_ids = zip(*chunks)
    input_ids = truncate(input_ids, max_length, value=0)
    attention_masks = truncate(attention_masks, max_length, value=1)
    patch_ids = truncate(patch_ids, max_length, value=0)

def message_parse(msg, tokenizer, max_length=56):
    msg = re.sub(r'#([0-9])+', '', msg)
    msg = re.sub(r'(\(|)([A-z])+-([0-9])+(\)|)(:|)', '', msg)
    msg = msg.strip()

    msg = tokenizer.tokenize(msg)
    msg = tokenizer.convert_tokens_to_ids(msg)
    msg = truncate(msg, max_length, value=0)


def job(sha_msgs, tokenizer):
    sha, msg = sha_msgs

    sha_parse(sha, tokenizer=tokenizer)
    message_parse(msg, tokenizer=tokenizer)

def main(args):
    sha_msgs = [(c.hexsha, c.summary) for c in repo.iter_commits()]
    func = partial(job, tokenizer=args.tokenizer)
    with Pool(processes=args.num_workers) as pool:
        pool.map(func, sha_msgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to collect commits on github")
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    args.local_path = args.url.split('/')[-1]
    logger.info(f"master branch of {args.url} will be downloaded to {args.local_path}")
    repo = (
        Repo(args.local_path)
        if os.path.exists(args.local_path)
        else Repo.clone_from(args.url, to_path=args.local_path, branch="master")
    )
    args.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")

    main(args)
