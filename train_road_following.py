import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from xy_dataset import XYDataset

lane = "left_lane_data"


# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


batch_sizes = [8, 16]
epochs = [100, 20, 50, 150]

for BATCH_SIZE in batch_sizes:
    for NUM_EPOCHS in epochs:

        # --- Configuration ---
        DATASET_DIR = f'{lane}'  # adjust if your dataset folder is different
        CATEGORIES = ['apex']
        LEARNING_RATE = 1e-4


        MODEL_OUTPUT_PATH = f'{lane}_e{BATCH_SIZE}_b{NUM_EPOCHS}.pth'


        # --- Transformations ---
        transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # --- Load Dataset ---
        dataset = XYDataset(directory=DATASET_DIR, categories=CATEGORIES, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # --- Model Setup ---
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Output: (x, y)
        model = model.to(device)

        # --- Loss and Optimizer ---
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Training Loop ---
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, _, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

        # --- Save Model ---
        torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
