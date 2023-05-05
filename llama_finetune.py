import argparse
import os
import pickle
import json
import pandas as pd

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, logging, set_seed, \
EarlyStoppingCallback


"""
Fine-Tune Llama-7b on SE paired dataset
"""

ROOT = "/gpfs/space/projects/stud_ml_22/NLP"
PATH_TO_CONVERTED_TOKENIZER = os.path.join(ROOT, "llama/7B_converted/")
PATH_TO_CONVERTED_WEIGHTS = os.path.join(ROOT, "llama/7B_converted/")
PATH_TO_DATASET = os.path.join(ROOT, "data/course_questions.pkl")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=PATH_TO_CONVERTED_WEIGHTS)
    parser.add_argument("--dataset_path", type=str, default=PATH_TO_DATASET)
    # parser.add_argument("--subset", type=str, default="data/finetune")
    # parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--frac_valid_set", type=float, default=0.05)
    # parser.add_argument("--streaming", action="store_true")
    # parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--load_8bit", action="store_true", default=False)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--run_name", default="", type=str)

    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), dataset.iterrows()), total=nb_examples):
        text = prepare_sample_text(example[1])
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['answer']}"
    return text


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else args.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = self.dataset.iterrows()
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(prepare_sample_text(next(iterator)[1]))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = self.dataset.iterrows()
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }


def create_datasets(tokenizer, args):
    # dataset = load_dataset(
    #     args.dataset_name,
    #     data_dir=args.subset,
    #     split=args.split,
    #     use_auth_token=True,
    #     num_proc=args.num_workers if not args.streaming else None,
    #     streaming=args.streaming,
    # )


    # if args.streaming:
    #     print("Loading the dataset in streaming mode")
    #     valid_data = dataset.take(args.size_valid_set)
    #     train_data = dataset.skip(args.size_valid_set)
    #     train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    # else:
    #     dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)
    #     train_data = dataset["train"]
    #     valid_data = dataset["test"]

    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f, encoding='utf8')

    data = data.reset_index(drop=True)

    valid_data = data.sample(frac=args.frac_valid_set, random_state=args.seed)
    train_data = data.drop(valid_data.index)
    valid_data = valid_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Log run parameters")
    d_args = vars(args)
    with open(os.path.join(args.output_dir, "finetune_config.json"), "w") as f:
        json.dump(d_args, f, indent=4)

    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    config = AutoConfig.from_pretrained(args.model_path)
    config.max_position_embeddings = args.seq_length
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        # use_cache=not args.no_gradient_checkpointing, # TODO: fix unexpected arg error
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_int8_training(model)

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=args.run_name,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)])

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    config = AutoConfig.from_pretrained(args.model_path)
    architecture = config.architectures[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)

    if "Llama" in architecture:
        print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
