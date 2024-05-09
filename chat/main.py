import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

hf_token = "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"

from transformers import AutoTokenizer

# 设置分词器和停止 ID
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# 使用HuggingFaceLLM
# generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

import torch
from llama_index.llms.huggingface import HuggingFaceLLM

# Optional quantization to 4bit
# import torch
# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-70B-Instruct",
    model_kwargs={
        "token": hf_token,
        "torch_dtype": torch.bfloat16,
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name="meta-llama/Meta-Llama-3-70B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
    device_map="auto",
)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    response = llm.complete("Who is Paul Graham?")

    print(response)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
