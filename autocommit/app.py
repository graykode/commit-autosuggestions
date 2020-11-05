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
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import (RobertaConfig, RobertaTokenizer)

from autocommit.model import Seq2Seq
from autocommit.utils import (Example, convert_examples_to_features)
from autocommit.model.diff_roberta import RobertaModel

from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

def get_model(model_class, config, tokenizer, mode):
    encoder = model_class(config=config)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
            beam_size=args.beam_size, max_length=args.max_target_length,
            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    assert args.load_model_path
    assert os.path.exists(os.path.join(args.load_model_path, mode, 'pytorch_model.bin'))

    model.load_state_dict(
        torch.load(
            os.path.join(args.load_model_path, mode, 'pytorch_model.bin'),
            map_location=torch.device('cpu')
        ),
        strict=False
    )
    return model

def get_features(examples):
    features = convert_examples_to_features(examples, args.tokenizer, args, stage='test')
    all_source_ids = torch.tensor(
        [f.source_ids[:args.max_source_length] for f in features], dtype=torch.long
    )
    all_source_mask = torch.tensor(
        [f.source_mask[:args.max_source_length] for f in features], dtype=torch.long
    )
    all_patch_ids = torch.tensor(
        [f.patch_ids[:args.max_source_length] for f in features], dtype=torch.long
    )
    return TensorDataset(all_source_ids, all_source_mask, all_patch_ids)

def create_app():
    @app.route('/')
    def index():
        return jsonify(hello="world")

    @app.route('/added', methods=['POST'])
    def added():
        if request.method == 'POST':
            payload = request.get_json()
            example = [
                Example(
                    idx=payload['idx'],
                    added=payload['added'],
                    deleted=payload['deleted'],
                    target=None
                )
            ]
            message = inference(model=args.added_model, data=get_features(example))
            return jsonify(idx=payload['idx'], message=message)

    @app.route('/diff', methods=['POST'])
    def diff():
        if request.method == 'POST':
            payload = request.get_json()
            example = [
                Example(
                    idx=payload['idx'],
                    added=payload['added'],
                    deleted=payload['deleted'],
                    target=None
                )
            ]
            message = inference(model=args.diff_model, data=get_features(example))
            return jsonify(idx=payload['idx'], message=message)

    @app.route('/tokenizer', methods=['POST'])
    def tokenizer():
        if request.method == 'POST':
            payload = request.get_json()
            tokens = args.tokenizer.tokenize(payload['code'])
            return jsonify(tokens=tokens)

    return app

def inference(model, data):
    # Calculate bleu
    eval_sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=len(data))

    model.eval()
    p=[]
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
                text = args.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    return p

def main(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    args.tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # budild model
    args.added_model =get_model(model_class=model_class, config=config,
                            tokenizer=args.tokenizer, mode='added').to(args.device)
    args.diff_model = get_model(model_class=model_class, config=config,
                            tokenizer=args.tokenizer, mode='diff').to(args.device)

    app = create_app()
    app.run(host=args.host, debug=True, port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--load_model_path", default='weight', type=str,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--config_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str,
                        default="microsoft/codebert-base", help="The name of tokenizer", )
    parser.add_argument("--max_source_length", default=512, type=int,
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

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    main(args)