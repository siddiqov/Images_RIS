import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm

# Define the path to your images
image_dir = 'E:/Oxford-pets/images'

# Load the Oxford Pets dataset from local directory
def load_dataset(image_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    return image_files

train_dataset = load_dataset(image_dir)

# Define image transformations for pre-processing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# 1. Feature Extraction using ResNet-50
def extract_features_resnet(images):
    model = models.resnet50(weights='DEFAULT')  # Use weights='DEFAULT'
    model.eval()
    features = []
    for img_path in tqdm(images):
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess_image(img)
        with torch.no_grad():
            feature = model(img_tensor.unsqueeze(0)).numpy().flatten()
            features.append(feature)
    return np.array(features)

# 2. Feature Extraction using CLIP

def extract_features_clip(images, processor, model):
    features = []
    for img_path in tqdm(images):
        if isinstance(img_path, str):  # File path
            img = Image.open(img_path).convert("RGB")
        else:  # PIL Image
            img = img_path.convert("RGB")
        
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feature = model.get_image_features(**inputs).numpy().flatten()
            features.append(feature)
    return np.array(features)


# Compare different techniques
def compare_techniques(query_img, images, feature_extractor, processor=None, model=None):
    # Check if query_img is a file path or PIL Image
    if isinstance(query_img, str):  # File path
        query_feature = feature_extractor([query_img], processor, model)[0]
    else:  # PIL Image
        query_feature = feature_extractor([query_img], processor, model)[0]
    
    # Ensure that images are always file paths for feature extraction
    if isinstance(images[0], str):  # File paths
        image_features = feature_extractor(images, processor, model)
    else:  # PIL Images
        image_features = feature_extractor([Image.open(img).convert("RGB") for img in images], processor, model)
    
    similarities = cosine_similarity([query_feature], image_features)
    return similarities[0]


# Example usage
if __name__ == "__main__":
    # Load sample images
    sample_images = train_dataset[:10]  # Taking the first 10 images

    # CLIP Processor and Model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # Load and preprocess query image
    query_image_path = 'E:/Oxford-pets/archive/images/images/Abyssinian_48.jpg'
    query_image = Image.open(query_image_path).convert("RGB")
    
    # Extract features and compare
    print("Comparing using ResNet-50...")
    similarities_resnet = compare_techniques(query_image_path, sample_images, extract_features_resnet)

    print("Comparing using CLIP...")
    similarities_clip = compare_techniques(query_image, sample_images, extract_features_clip, clip_processor, clip_model)

    # Print top-5 similar images
    print("Top-5 most similar images using ResNet-50:")
    top5_resnet = np.argsort(similarities_resnet)[-5:]
    print(top5_resnet)

    print("Top-5 most similar images using CLIP:")
    top5_clip = np.argsort(similarities_clip)[-5:]
    print(top5_clip)


