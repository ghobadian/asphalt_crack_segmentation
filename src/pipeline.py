# ===================================================================
# 4. RUNNER SCRIPT
# ===================================================================

print("--- Setting up for training ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the model
crack_model = SimpleUNet(in_channels=3, out_channels=1).to(device)

# --- Start or Resume Training ---
train_model(
    model=crack_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    epochs=50
)

# --- Start Evaluation ---
print("\n--- Loading best model for evaluation ---")
eval_model = SimpleUNet(in_channels=3, out_channels=1).to(device)
best_model_path = os.path.join(MODELS_DIR, 'best_model.pth')

if os.path.exists(best_model_path):
    eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
    evaluate_model(eval_model, test_loader, device)
    visualize_predictions(eval_model, test_loader, device, RESULTS_DIR)
else:
    print("Could not find best_model.pth. Skipping evaluation.")