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
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import (RobertaConfig, RobertaTokenizer)

import argparse
import whatthepatch
from train.run import (Example, convert_examples_to_features)
from train.model import Seq2Seq
from train.customized_roberta import RobertaModel

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def create_examples(diff, tokenizer):
    examples = []
    for idx, example in enumerate(whatthepatch.parse_patch(diff)):
        added, deleted = [], []
        for change in example.changes:
            if change.old == None and change.new != None:
                added.extend(tokenizer.tokenize(change.line))
            elif change.old != None and change.new == None:
                deleted.extend(tokenizer.tokenize(change.line))
        examples.append(
            Example(
                idx=idx,
                added=added,
                deleted=deleted,
                target=None
            )
        )

    return examples

def main(args):

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # budild model
    encoder = model_class(config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path), strict=False)

    model.to(args.device)
    with open("test.source", "r") as f:
        eval_examples = create_examples(f.read(), tokenizer)

    test_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
    all_patch_ids = torch.tensor([f.patch_ids for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_source_ids, all_source_mask, all_patch_ids)

    # Calculate bleu
    eval_sampler = SequentialSampler(test_data)
    eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=len(test_data))

    model.eval()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, patch_ids = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask, patch_ids=patch_ids)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--load_model_path", default=None, type=str, required=True,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--config_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str,
                        default="microsoft/codebert-base", help="The name of tokenizer", )
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    main(args)