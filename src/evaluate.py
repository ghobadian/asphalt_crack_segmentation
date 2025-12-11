# ===================================================================
# 3. EVALUATION LOGIC
# ===================================================================

def iou_score(pred, target, smooth=1e-5):
    """Calculates the Intersection over Union (IoU) score."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def visualize_predictions(model, loader, device, results_dir, num_images=5):
    """Saves visualizations of model predictions."""
    print(f"Visualizing {num_images} predictions...")
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = torch.sigmoid(model(images))

    images = images.cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.cpu().squeeze(1).numpy()
    preds = outputs.cpu().squeeze(1).numpy() > 0.5

    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=16)
    for i in range(num_images):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(preds[i], cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = os.path.join(results_dir, 'test_predictions.png')
    plt.savefig(save_path)
    print(f"🖼️ Prediction visualization saved to {save_path}")
    plt.show()

def evaluate_model(model, loader, device):
    """Evaluates the model on the test set using the IoU metric."""
    print("🧪 Starting evaluation...")
    model.eval()
    total_iou = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="[Evaluating]"):
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            batch_iou = iou_score(outputs, masks)
            total_iou += batch_iou.item()
    avg_iou = total_iou / len(loader)
    print(f"\n--- Evaluation Complete ---")
    print(f"📊 Average Intersection over Union (IoU): {avg_iou:.4f}")
    return avg_iou