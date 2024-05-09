import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    # device="auto",
    # device=torch.device("auto" if torch.cuda.is_available() else "cpu"),
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a cute cat"},
    {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "What can you do?"},
    {"role": "user", "content": "Where are you from?"},
    # {"role": "user", "content": "How old are you?"},
    # {"role": "user", "content": "Fuck you"},
    # {"role": "user", "content": "Do you have children"},
    # {"role": "user", "content": "Can you tell me a joke?"},
    # {"role": "user", "content": "Can you speak Chinese?"},
    # {"role": "user", "content": "Do you know ShangHai and Beijing?"},
    # {"role": "user", "content": "你的下一个目的地是哪里?"},

]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# print(outputs[0]["generated_text"])
print(outputs[0]["generated_text"][len(prompt):])
