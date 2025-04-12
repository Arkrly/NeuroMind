# Import the CustomCNN class
from custom_cnn_module import CustomCNN  # Replace 'custom_cnn_module' with the actual module name
import torch  # Import the torch module

# Load the best model for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define the device
model = CustomCNN().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  