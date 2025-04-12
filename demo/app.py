import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet18 model
def get_resnet18_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
    return model.to(device)

# Define original CustomCNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def get_custom_cnn_model():
    return CustomCNN().to(device)

# Load the trained model with diagnostics
def load_model(model_path="best_model.pth"):
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle potential 'state_dict' wrapping
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Try ResNet18 first
    resnet_model = get_resnet18_model()
    try:
        resnet_model.load_state_dict(state_dict)
        resnet_model.eval()
        st.success("Model loaded successfully as ResNet18.")
        return resnet_model
    except RuntimeError as e:
        st.warning(f"Failed to load as ResNet18: {str(e)}")

    # Fallback to CustomCNN
    custom_model = get_custom_cnn_model()
    try:
        custom_model.load_state_dict(state_dict)
        custom_model.eval()
        st.success("Model loaded successfully as CustomCNN.")
        return custom_model
    except RuntimeError as e:
        st.error(f"Failed to load as CustomCNN: {str(e)}")
        st.write("State dict keys in file:", list(state_dict.keys())[:5])  # Show first 5 keys for debugging
        return None

# Load the model
model = load_model()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
class_labels = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Streamlit UI
st.title("Alzheimer's Disease Classification")
st.write("Upload an image to classify the stage of Alzheimer's disease.")

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
            predicted_label = class_labels[predicted_class.item()]

        # Display result
        st.write(f"**Predicted Class:** {predicted_label}")
else:
    st.write("Model loading failed. Please check the error messages above and ensure 'best_model.pth' matches one of the defined architectures.")