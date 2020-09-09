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

import torch
import argparse
import subprocess
from transformers import AutoTokenizer

from preprocess import diff_parse, truncate
from train import BartForConditionalGeneration


def suggester(chunks, max_source_length, model, tokenizer, device):
    input_ids, attention_masks, patch_ids = zip(*chunks)
    input_ids = torch.LongTensor([truncate(input_ids, max_source_length, value=0)]).to(
        device
    )
    attention_masks = torch.LongTensor(
        [truncate(attention_masks, max_source_length, value=1)]
    ).to(device)
    patch_ids = torch.LongTensor([truncate(patch_ids, max_source_length, value=0)]).to(
        device
    )

    summaries = model.generate(
        input_ids=input_ids, patch_ids=patch_ids, attention_mask=attention_masks
    )
    return tokenizer.batch_decode(
        summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model = BartForConditionalGeneration.from_pretrained(args.output_dir).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if args.unittest:
        with open("test.source", "r") as f:
            chunks = diff_parse(f.read(), tokenizer)
    else:
        proc = subprocess.Popen(["git", "diff", "--cached"], stdout=subprocess.PIPE)
        staged_files = proc.stdout.readlines()
        staged_files = [f.decode("utf-8") for f in staged_files]
        staged_files = [f.strip() for f in staged_files]
        chunks = "\n".join(staged_files)

    commit_message = suggester(
        chunks,
        max_source_length=args.max_source_length,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(commit_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to collect commits on github")
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--unittest", action="store_true", help="Unittest with an one batch git diff"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="sshleifer/distilbart-xsum-6-6",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    args = parser.parse_args()

    main(args)
