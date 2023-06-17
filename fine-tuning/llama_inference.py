import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from accelerate import Accelerator
import time

import os

ROOT = "/gpfs/space/projects/stud_ml_22/NLP"

PATH_TO_CONVERTED_WEIGHTS = os.path.join(ROOT, "llama/7B_converted/")
PATH_TO_CONVERTED_TOKENIZER = os.path.join(ROOT, "llama/7B_converted/")
PATH_TO_ADAPTER = os.path.join(ROOT, "experiments/debug_a100/final_checkpoint")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

peft_config = PeftConfig.from_pretrained(PATH_TO_ADAPTER)

config = AutoConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
config.max_position_embeddings = 1024

model = AutoModelForCausalLM.from_pretrained(
        PATH_TO_CONVERTED_WEIGHTS,
        config=config,
        trust_remote_code=True,
        # use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
        # device_map="auto"
    )

model = PeftModel.from_pretrained(model, PATH_TO_ADAPTER, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

try:
    with torch.no_grad():
        while True:
            prompt = input("Me: ")
            s = time.time()

            inputs = tokenizer(prompt, return_tensors="pt")

            e1 = time.time()

            generate_ids = model.generate(input_ids=inputs.input_ids.to(device), max_length=100) # max_length = max_new_tokens + prompt_length

            e2 = time.time()
            print('Llama: ', end='')
            print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

            e3 = time.time()
            # print the times in %H:%M:%S format
            print("Time to tokenize: ", time.strftime('%H:%M:%S', time.gmtime(e1 - s)))
            print("Time to generate: ", time.strftime('%H:%M:%S', time.gmtime(e2 - e1)))
            print("Time to decode: ", time.strftime('%H:%M:%S', time.gmtime(e3 - e2)))

except KeyboardInterrupt:
    exit(0)
