## How the model was trained.
We used the pre-trained weight provided by CodeBERT(Feng at al, 2020) as the initial weight.

#### Added model
To train the added model, you can train it using [CodeBERT's official repository](https://github.com/microsoft/CodeBERT). For training data, the cleaned CodeSearchNet was used. See [this document](https://github.com/microsoft/CodeBERT#fine-tune-1) for details. I took about 23 hours with 256 batch size.

```shell script
cd code2nl

lang=python #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../data/code2nl/CodeSearchNet
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
```

#### Diff model
To train the Diff model we have to use [our code](https://github.com/graykode/commit-autosuggestions/blob/master/train.py). We need an implementation to differentiate between added and diff.
As for the training data, only the top 100 repositories of the Python language in [the document](https://github.com/kaxap/arl/blob/master/README-Python.md) were cloned ([gitcloner.py](https://github.com/graykode/commit-autosuggestions/blob/master/gitparser.py)), and the commit message, added and deleted were preprocessed in jsonl format ([gitparser](https://github.com/graykode/commit-autosuggestions/blob/master/gitparser.py)). The data we used was put on a [google drive](https://drive.google.com/drive/folders/1_8lQmzTH95Nc-4MKd1RP3x4BVc8tBA6W?usp=sharing).
Like the added model, it took about 20 hours at 256 batch size for training.
Note that the weight of the added model was used as the initial weight. Be sure to set this with the `load_model_path` argument.

```shell script
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
output_dir=model/python
train_file=train.jsonl
dev_file=valid.jsonl

eval_steps=1000
train_steps=50000
saved_model=pytorch_model.bin # this is added model weight

python train.py --do_train --do_eval --model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--load_model_path $saved_model \
	--train_filename $train_file \
	--dev_filename $dev_file \
	--output_dir $output_dir \
	--max_source_length $source_length \
	--max_target_length $target_length \
	--beam_size $beam_size \
	--train_batch_size $batch_size \
	--eval_batch_size $batch_size \
	--learning_rate $lr \
	--train_steps $train_steps \
	--eval_steps $eval_steps
```

## How to train for your lint style?
See the [Diff model](https://github.com/graykode/commit-autosuggestions/blob/master/docs/training.md#diff-model) section above for the role of the code.

#### 1. cloning repositories from github
This code clones all repositories in [repositories.txt](https://github.com/graykode/commit-autosuggestions/blob/master/repositories.txt).
```shell script
usage: gitcloner.py [-h] --repositories REPOSITORIES --repos_dir REPOS_DIR [--num_worker_threads NUM_WORKER_THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --repositories REPOSITORIES
                        repositories file path.
  --repos_dir REPOS_DIR
                        directory that all repositories will be downloaded.
  --num_worker_threads NUM_WORKER_THREADS
                        number of threads in a worker

```

#### 2. parsing added code, deleted code and commit message from cloned repositories.
This code preprocesses cloned repositories and divides them into train, valid, and test data.

```shell script
usage: gitparser.py [-h] --repositories REPOSITORIES --repos_dir REPOS_DIR --output_dir OUTPUT_DIR [--tokenizer_name TOKENIZER_NAME] [--num_workers NUM_WORKERS]
                    [--max_source_length MAX_SOURCE_LENGTH] [--max_target_length MAX_TARGET_LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --repositories REPOSITORIES
                        repositories file path.
  --repos_dir REPOS_DIR
                        directory that all repositories had been downloaded.
  --output_dir OUTPUT_DIR
                        The output directory where the preprocessed data will be written.
  --tokenizer_name TOKENIZER_NAME
                        The name of tokenizer
  --num_workers NUM_WORKERS
                        number of process
  --max_source_length MAX_SOURCE_LENGTH
                        The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
  --max_target_length MAX_TARGET_LENGTH
                        The maximum total target sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
```

> If `UnicodeDecodeError` occurs while using gitparser.py, you must use the [GitPython](https://github.com/gitpython-developers/GitPython) package at least [this commit](https://github.com/gitpython-developers/GitPython/commit/bfbd5ece215dea328c3c6c4cba31225caa66ae9a).

#### 3. Training Added model(Optional for Python Language).
Python has learned the Added model. So, if you only want to make a Diff model for the Python language, step 3 can be ignored. However, for other languages (JavaScript, GO, Ruby, PHP and JAVA), [Code2NL training](https://github.com/microsoft/CodeBERT#fine-tune-1) is required to use as the initial weight of the model to be used in step 4.

#### 4. Training Diff model.
Train the Diff model as the initial weight of the added model for each languages.

```shell script
usage: train.py [-h] --model_type MODEL_TYPE --model_name_or_path MODEL_NAME_OR_PATH --output_dir OUTPUT_DIR [--load_model_path LOAD_MODEL_PATH]
                [--train_filename TRAIN_FILENAME] [--dev_filename DEV_FILENAME] [--test_filename TEST_FILENAME] [--config_name CONFIG_NAME] [--tokenizer_name TOKENIZER_NAME]
                [--max_source_length MAX_SOURCE_LENGTH] [--max_target_length MAX_TARGET_LENGTH] [--do_train] [--do_eval] [--do_test] [--do_lower_case] [--no_cuda]
                [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--learning_rate LEARNING_RATE] [--beam_size BEAM_SIZE] [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
                [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS] [--eval_steps EVAL_STEPS] [--train_steps TRAIN_STEPS] [--warmup_steps WARMUP_STEPS]
                [--local_rank LOCAL_RANK] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        Model type: e.g. roberta
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model: e.g. roberta-base
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --load_model_path LOAD_MODEL_PATH
                        Path to trained model: Should contain the .bin files
  --train_filename TRAIN_FILENAME
                        The train filename. Should contain the .jsonl files for this task.
  --dev_filename DEV_FILENAME
                        The dev filename. Should contain the .jsonl files for this task.
  --test_filename TEST_FILENAME
                        The test filename. Should contain the .jsonl files for this task.
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name
  --max_source_length MAX_SOURCE_LENGTH
                        The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
  --max_target_length MAX_TARGET_LENGTH
                        The maximum total target sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --do_test             Whether to run eval on the dev set.
  --do_lower_case       Set this flag if you are using an uncased model.
  --no_cuda             Avoid using CUDA when available
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --beam_size BEAM_SIZE
                        beam size for beam search
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform. Override num_train_epochs.
  --eval_steps EVAL_STEPS
  --train_steps TRAIN_STEPS
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --seed SEED           random seed for initialization
```