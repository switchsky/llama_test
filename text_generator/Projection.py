
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomModelForBinaryClassification(nn.Module):
    def __init__(self, model_name):
        super(CustomModelForBinaryClassification, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.linear = nn.Linear(self.model.config.hidden_size, 20)
        self.batch_norm = nn.BatchNorm1d(self.model.config.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(20,2) # 2个神经元用于二分类
        # 冻结 LLaMA-3 模型的参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last hidden states
        pooled_output = torch.mean(hidden_states, dim=1)  # Pool the hidden states
        normalized_output = self.batch_norm(pooled_output)
        proto = self.linear(normalized_output)
        proto = self.relu(proto)
        logits = self.classifier(proto)
        return logits

if __name__ == '__main__':
    print(111)