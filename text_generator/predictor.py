import os

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from EmoDataset import EmoDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from text_generator.Projection import CustomModelForBinaryClassification
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import transformers
import torch


def load_model(model_path, model_name):
    model = CustomModelForBinaryClassification(model_name)
    checkpoint = torch.load(model_path)
    model.linear.load_state_dict(checkpoint['linear'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    return model

if __name__ == '__main__':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_path = "../pth/model_epoch2.pth"  # 修改为你的实际路径
    model = load_model(model_path, model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    with torch.no_grad():
        for i in range (100):
            text = "Describe the emotion of this text in one word:"
            critic = input("Please enter your critic: ")
            text = text + critic + tokenizer.eos_token
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            print(inputs.convert_to_tensors())
            input_ids = inputs['input_ids'].squeeze(1)
            attention_mask = inputs['attention_mask'].squeeze(1)
            print(attention_mask)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(outputs)