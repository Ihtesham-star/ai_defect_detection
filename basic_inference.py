"""
Basic Inference Script - Open Source Version
===========================================

This is the FREE version. Requires your own trained model.

For production features with pre-trained model:
- Pre-trained model (70K images): $299 one-time OR $49/month
- Hosted solution: $99-299/month
- Contact: your.email@example.com
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys

class MultiDomainDefectCNN(nn.Module):
    """Model architecture - matches the trained production model"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(MultiDomainDefectCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Dropout2d(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def predict_image(model_path, image_path, device='cpu'):
    """
    Basic inference function
    
    Args:
        model_path: Path to your trained .pth model
        image_path: Path to image for analysis
        device: 'cpu' or 'cuda'
    """
    
    # Load model
    model = MultiDomainDefectCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu()
        confidence, predicted = torch.max(probabilities, 1)
    
    # Results
    class_names = ['DEFECTIVE', 'GOOD']
    result = {
        'prediction': class_names[predicted.item()],
        'confidence': confidence.item(),
        'defect_probability': probabilities[0][0].item(),
        'good_probability': probabilities[0][1].item()
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Basic AI Defect Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to your trained model (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image for analysis')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Check if model exists
    try:
        result = predict_image(args.model, args.image, args.device)
        
        print("\n" + "="*50)
        print("AI DEFECT DETECTION RESULTS")
        print("="*50)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Defect Probability: {result['defect_probability']:.1%}")
        print(f"Good Probability: {result['good_probability']:.1%}")
        print("="*50)
        
    except FileNotFoundError as e:
        print("\n❌ ERROR: Model file not found!")
        print("\nYou need a trained model to use this tool.")
        print("\nOptions:")
        print("1. Train your own model (see docs/TRAINING.md)")
        print("2. Get our pre-trained model:")
        print("   - One-time: $299")
        print("   - Monthly: $49/month")
        print("   - Contact: your.email@example.com")
        print("\n3. Use our hosted solution (no setup): $99-299/month")
        print("   - Visit: https://yourwebsite.com")
        sys.exit(1)


if __name__ == "__main__":
    main()