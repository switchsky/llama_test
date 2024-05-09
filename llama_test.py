#修改为镜像环境
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
llama_version_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_version_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
llama_version = llama_version_70B
#设置huggingface秘钥
hf_token = "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"
# 设置分词器和停止 ID
tokenizer = AutoTokenizer.from_pretrained(
    llama_version,
    token=hf_token
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# 使用HuggingFaceLLM
# generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

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
    model_name=llama_version,
    model_kwargs={
        "token": hf_token,
        "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name=llama_version,
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
    device_map="auto"
)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    response = llm.complete("can you tell me federated learning")
    print(response)
