#修改为镜像环境
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from tqdm import tqdm
import transformers
from accelerate import infer_auto_device_map

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:4",
)

pic_dir = "../data/pic/"
results = []
# prepare image and text prompt, using the appropriate prompt template
# Loop through the images and process each one
for i in tqdm(range(1, 3),desc="Processing images"):
    # Load the image
    image = Image.open(pic_dir + str(i) + ".png")

    # Prepare image and text prompt using the appropriate prompt template
    prompt = ("[INST] <image>\n"
              "Describe the satellite image in detail - Provide a detailed description of the geographical features in the image - Offer a comprehensive summary of human activity, urban infrastructure, and environments in aerial image. Unambiguous expressions are prohibited, providing clear and accurate descriptions.\n[/INST]")

    # Process the input
    inputs = processor(prompt, image, return_tensors="pt").to('cuda:4')

    # Generate the output
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.5, do_sample=True)

    # Decode the output and store the result
    result = processor.decode(output[0], skip_special_tokens=True)
    results.append(result)

# Print or process the results as needed
for i, result in enumerate(results):
    print(f"Result for image {i + 1}:\n{result}\n")


