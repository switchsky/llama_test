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
import prompt as pt
import yaml

areas_sorted_bynum = pt.select_topK("../data/poi/all_area_counts_num.txt", 3, 0)
areas_sorted_byprop = pt.select_topK("../data/poi/all_area_counts_prop.txt", 3, method=1)

# 区域编号，top_k
def prompt_all(area):
    base = "[INST] <image>\n""[/INST]"
    prompt_poi = pt.prompt_generate(areas_sorted_bynum[area], areas_sorted_byprop[area])
    prompt = "[INST] <image>\n"+prompt_poi+"[/INST]"
    return prompt

def extract_result(text):
    start_token = "[/INST]"
    start_index = text.find(start_token) + len(start_token)
    result = text[start_index:].strip()
    return result

def save(result,area):
    url = "../data/entity/text/"+str(area)+".txt"
    with open(url, 'w') as f:
        f.write(result)

if __name__ == '__main__':
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:5",
    )

    pic_dir = "../data/entity/img/"
    results = []
    # prepare image and text prompt, using the appropriate prompt template
    # Loop through the images and process each one
    # 先读取配置文件，查看写到第几轮/默认为0
    with open('../data/entity/text/config.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    complete = int(result['complete'])
    for i in tqdm(range(complete + 1, 677),desc="Processing images"):
        # Load the image
        image = Image.open(pic_dir + str(i) + ".png")
        # Prepare image and text prompt using the appropriate prompt template
        prompt = (prompt_all(i-1))

        # Process the input
        inputs = processor(prompt, image, return_tensors="pt").to('cuda:5')

        # Generate the output
        output = model.generate(**inputs, max_new_tokens=512, temperature=0.8, top_p = 0.9, do_sample=True)

        # Decode the output and store the result
        result = processor.decode(output[0], skip_special_tokens=True)
        result = extract_result(result)
        save(result,i)
        areaData = {"complete": i}
        with open('../data/entity/text/config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data=areaData, stream=f, allow_unicode=True)


