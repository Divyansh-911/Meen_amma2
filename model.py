import torch
from torchvision import transforms
from PIL import Image

# Simulated class dictionary
class_map = {
    0: {"name": "Bacterial Gill Disease", "cause": "Gills rot due to bacteria, causing respiratory issues."},
    1: {"name": "Ichthyophthiriasis (White Spot)", "cause": "Parasitic infection, causes white spots on skin."},
    2: {"name": "Saprolegniasis", "cause": "Fungal infection with cottony growth on skin or gills."},
    3: {"name": "Viral Hemorrhagic Septicemia", "cause": "Internal bleeding due to virus, highly contagious."},
    4: {"name": "Swim Bladder Disorder", "cause": "Genetic or bacterial, affects buoyancy control."}
}

def predict_disease(image: Image.Image) -> dict:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)
    class_idx = torch.randint(0, len(class_map), (1,)).item()
    result = class_map[class_idx]
    return {
        "disease": result["name"],
        "effect": result["cause"]
    }
