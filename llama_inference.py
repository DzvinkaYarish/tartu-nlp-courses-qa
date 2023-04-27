from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from accelerate import Accelerator

PATH_TO_CONVERTED_WEIGHTS = ""
PATH_TO_CONVERTED_TOKENIZER = ""

# model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

model = AutoModelForCausalLM.from_pretrained(
        PATH_TO_CONVERTED_WEIGHTS,
        trust_remote_code=True,
        # use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
    )

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

try:
    while True:
        prompt = input("Me: ")

        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate

        generate_ids = model.generate(inputs.input_ids, max_length=30)

        print('Llama: ', end='')
        print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
except KeyboardInterrupt:
    exit(0)
