import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import os
import gdown

# Define your CNNModel class matching the original architecture
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        
        # Load the ResNet-18 model with the custom final layer
        self.resnet = models.resnet18(pretrained=False)  # Ensure this matches your training setup
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Google Drive file ID
GDRIVE_FILE_ID = '170Hh8CJGr0u2oJEKtimcCJyVa_JjRHiD'

# Download the model directly from Google Drive
@st.cache_resource  # Cache the download and loading of the model
def load_model_from_gdrive():
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    output = 'temp_model.pth.tar'  # Temporary file name for the downloaded model
    gdown.download(url, output, quiet=False)
    
    model = CNNModel(num_classes=5)  # Instantiate your model
    
    # Load the state dictionary into the model
    state_dict = torch.load(output, map_location=torch.device('cpu'))
    
    # Rename keys by adding 'resnet.' prefix
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = f"resnet.{key}"
        new_state_dict[new_key] = state_dict[key]
    
    # Load the updated state dictionary
    model.load_state_dict(new_state_dict)
    model.eval()  # Set the model to evaluation mode
    
    os.remove(output)  # Remove the temporary model file after loading
    return model

# Define the class names (adjust based on your dataset)
class_names = ['Early Blight', 'Healthy', 'Late Blight', 'Septoria leaf spot', 'Tomato mosaic virus']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Tomato Leaf Disease Detection")
st.write("Upload an image of a tomato leaf to predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Transform the image and make a prediction
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        model = load_model_from_gdrive()  # Load the model
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = class_names[predicted]

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence Score:** {confidence.item():.2f}")
