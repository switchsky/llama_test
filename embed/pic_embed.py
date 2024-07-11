import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import ViTModel, ViTConfig
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



if __name__ == '__main__':
    # Step 1: Load the pre-trained ViT model
    model_name = "google/vit-base-patch16-224"
    model = ViTModel.from_pretrained(model_name).cuda()

    # Step 2: Define the image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    features_list = {}
    # Step 3: Load and preprocess an image
    image_dir = "../data/entity/img/"
    out_path = "../data/entity/embed/visual_embed/visual_features_ent.pt"
    for image_number in tqdm(range(0,3), desc="Image embeding"):
        image_path = os.path.join(image_dir, str(image_number+1))
        image_path += ".png"
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0).cuda()  # Add batch dimension
        # Step 4: Use the model's embeddings
        with torch.no_grad():
            outputs = model(pixel_values=image)
            embeddings = outputs.last_hidden_state  # or outputs.pooler_output for pooled embeddings
            embeddings = outputs.pooler_output
        features_list[image_number] = embeddings
        # print(features_list)
    torch.save(features_list, out_path)
        # Now `embeddings` contains the embeddings of the input image