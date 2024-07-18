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
def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, return_tensors="pt")

def collate_fn(batch, tokenizer,max_length=512):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    #print(texts[1])
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    encodings['labels'] = torch.tensor(labels)
    return encodings

if __name__ == '__main__':
    # 定义数据目录
    positive_dir = '../data/emotion/pos'
    negative_dir = '../data/emotion/neg'
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = CustomModelForBinaryClassification(model_id)
    # 交叉熵损失 b*t batchsize和第t类别
    criterion = nn.CrossEntropyLoss()

    EmoDataset = EmoDataset(positive_dir, negative_dir)
    train_dataloader = DataLoader(EmoDataset, shuffle=True, batch_size=16,collate_fn=lambda batch: collate_fn(batch, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * 3  # 3个epoch
    scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    total_loss = 0
    step = 0
    for epoch in range(3):
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels = batch['labels']
            print(labels)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            step += 1
            # 打印每一步的损失
            print(f"Step {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            if step % 30 ==0 and step != 0:
                torch.save({
                    'linear': model.linear.state_dict(),
                    'classifier': model.classifier.state_dict()
                }, f"../pth/model_epoch_step{step}.pth")
    # x = torch.tensor(
    #     [[1,0.3]]
    # )
    # y = torch.tensor([1])
    # print(criterion(x,y))