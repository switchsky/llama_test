#修改为镜像环境
import os

import transformers
from accelerate import infer_auto_device_map

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model = LlavaNextForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     device_map="cuda:2"
# )
pipeline = transformers.pipeline(
    "image-to-text",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:4"
)
pic_dir = "../data/pic/"
images = []
# prepare image and text prompt, using the appropriate prompt template
for i in range(1,3):
    image = Image.open(pic_dir + str(i) + ".png")
    images.append(image)
prompt = ("[INST] <image>\n"
          "Describe the satellite image in detail - Provide a detailed description of the geographical features in the image - Offer a comprehensive summary of human activity, urban infrastructure, and environments in aerial image,Unambiguous expressions are prohibited, providing clear and accurate descriptions.\n[/INST]")

inputs = processor(prompt, image, return_tensors="pt")

# autoregressively complete prompt
#output = model.generate(**inputs, max_new_tokens=100)
outputs = pipeline(
    prompt = prompt,
    max_new_tokens = 400,
    images = images,
)
print(outputs[0][0]['generated_text'])
print("-------------------")
print(outputs[1][0]['generated_text'])
