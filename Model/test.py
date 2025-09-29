import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image

# ==============================================================================
# --- YOUR MODEL ARCHITECTURE ---
# ==============================================================================
# This is the exact CNN class you provided.
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176) # Flatten
        out = self.dense_layers(out) # Fully connected
        return out

# ==============================================================================
# --- USER MUST MODIFY THIS SECTION ---
# ==============================================================================

# 1. DEFINE ALL CLASS NAMES (Updated with Mendeley Dataset classes)
#    This list MUST match the output classes of your model in the correct order.
#    Total classes = 39.
CLASS_NAMES = [
    'Apple_scab',
    'Apple_black_rot',
    'Apple_cedar_apple_rust',
    'Apple_healthy',
    'Background_without_leaves',
    'Blueberry_healthy',
    'Cherry_powdery_mildew',
    'Cherry_healthy',
    'Corn_gray_leaf_spot',
    'Corn_common_rust',
    'Corn_northern_leaf_blight',
    'Corn_healthy',
    'Grape_black_rot',
    'Grape_black_measles',
    'Grape_leaf_blight',
    'Grape_healthy',
    'Orange_haunglongbing',
    'Peach_bacterial_spot',
    'Peach_healthy',
    'Pepper_bacterial_spot',
    'Pepper_healthy',
    'Potato_early_blight',
    'Potato_healthy',
    'Potato_late_blight',
    'Raspberry_healthy',
    'Soybean_healthy',
    'Squash_powdery_mildew',
    'Strawberry_healthy',
    'Strawberry_leaf_scorch',
    'Tomato_bacterial_spot',
    'Tomato_early_blight',
    'Tomato_healthy',
    'Tomato_late_blight',
    'Tomato_leaf_mold',
    'Tomato_septoria_leaf_spot',
    'Tomato_spider_mites_two-spotted_spider_mite',
    'Tomato_target_spot',
    'Tomato_mosaic_virus',
    'Tomato_yellow_leaf_curl_virus'
]

# 2. DEFINE PATHS
#    Update these paths to point to your model file and the image you want to test.
MODEL_PATH = 'plant_disease_model_1_latest.pt'  # <-- IMPORTANT: Change this!
IMAGE_PATH = '/home/vasu/Plant-Disease-Detection/test_images/corn_northen_leaf_blight.JPG'     # <-- IMPORTANT: Change this!

# ==============================================================================
# --- END OF MODIFICATION SECTION ---
# ==============================================================================


def predict(image_path, model, class_names, device):
    """
    Takes an image path and a trained model, preprocesses the image,
    and returns the predicted plant/disease and the model's confidence.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}", None

    image = image.resize((224, 224))
    input_tensor = TF.to_tensor(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, index = torch.max(probabilities, 0)
    predicted_class = class_names[index.item()]

    # --- Updated Formatting Logic ---
    if predicted_class == 'Background_without_leaves':
        plant = 'N/A'
        disease = 'Background Image'
    else:
        # Splits 'Apple_cedar_apple_rust' into 'Apple' and 'cedar_apple_rust'
        parts = predicted_class.split('_', 1)
        plant = parts[0]
        disease = parts[1].replace('_', ' ') # Replaces underscores with spaces

    return f"Plant: {plant}, Disease: {disease}", confidence.item()


if __name__ == '__main__':
    # Verify the number of classes
    num_classes = len(CLASS_NAMES)
    if num_classes != 39:
        print(f"[Warning] The CLASS_NAMES list has {num_classes} items, but 39 were expected.")
    
    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate your custom model with the correct number of output classes (K)
    model = CNN(K=num_classes)

    # Load the saved model weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
    except FileNotFoundError:
        print(f"\n[ERROR] Model weights file not found at '{MODEL_PATH}'")
        print("Please update the 'MODEL_PATH' variable in the script.")
        exit()
    except Exception as e:
        print(f"\n[ERROR] Failed to load the model: {e}")
        print("Please ensure your saved '.pt' file was trained with the exact same CNN architecture "
              f"and has an output layer for {num_classes} classes.")
        exit()

    # Perform and print the prediction
    prediction, confidence = predict(IMAGE_PATH, model, CLASS_NAMES, device)
    if prediction:
        print(f"\n--- Prediction Result ---")
        print(f"Image Path: '{IMAGE_PATH}'")
        print(f"Prediction:   {prediction}")
        print(f"Confidence:   {confidence * 100:.2f}%")