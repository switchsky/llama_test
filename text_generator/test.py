import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # 加载模型和tokenizer
    model = load_model(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    with torch.no_grad():
        for i in range(100):
            text = "Please determine the sentiment of this text, and Keep the output as short as possible:"
            critic = input("Please enter your critic: ")
            text = text + " " + critic + " " + tokenizer.eos_token

            # Tokenize the input text
            model_inputs = tokenizer([text], return_tensors='pt')
            print(model_inputs)

            # Forward pass to get the model outputs
            outputs = model(model_inputs.input_ids)

            # Extract logits from the outputs
            logits = outputs.logits

            # Get the predicted token IDs by taking the argmax of the logits
            predicted_token_ids = torch.argmax(logits, dim=-1)

            # Decode the token IDs to get the generated text
            predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

            print(predicted_text)