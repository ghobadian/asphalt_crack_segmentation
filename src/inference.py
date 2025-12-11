# src/inference.py
"""
Script for running inference on new images.
"""

import torch
import cv2
import numpy as np
from model_architecture import SimpleUNet


def predict_single_image(model_path, image_path, output_path, img_size=512, device='cuda'):
    """
    Run inference on a single image.

    Args:
        model_path: Path to the trained model weights
        image_path: Path to the input image
        output_path: Path to save the prediction mask
        img_size: Model input size
        device: Device to run inference on
    """
    # Load model
    model = SimpleUNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    image_resized = cv2.resize(image, (img_size, img_size))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
        prediction = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Resize back to original size
    prediction_resized = cv2.resize(prediction, (original_size[1], original_size[0]))

    # Save result
    cv2.imwrite(output_path, prediction_resized)
    print(f"Prediction saved to {output_path}")

    return prediction_resized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crack Segmentation Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='prediction.png', help='Output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    predict_single_image(args.model, args.image, args.output, device=args.device)