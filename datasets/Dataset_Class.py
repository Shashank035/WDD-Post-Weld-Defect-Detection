import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

class WeldDefectDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, 'images'))
        self.labels = os.listdir(os.path.join(root_dir, 'labels'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.images[idx])
        label_path = os.path.join(self.root_dir, 'labels', self.labels[idx])
        
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Load and process label data here
        # For simplicity, assume labels are in YOLO format
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        # Convert labels to tensors
        labels = [line.strip().split() for line in labels]
        labels = [[float(x) for x in label] for label in labels]
        
        return image, torch.tensor(labels)

class YOLOModel(nn.Module):
    def __init__(self):
        super(YOLOModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*16*16, 128)  # Assuming input size is 64x64
        self.fc2 = nn.Linear(128, 7)  # Output layer for bounding box info

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256*16*16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    
    # Find the maximum number of bounding boxes across all labels
    max_boxes = max(label.shape[0] for label in labels)
    
    # Pad each label to have max_boxes
    padded_labels = []
    for label in labels:
        padded_label = torch.zeros((max_boxes, label.shape[1]))
        padded_label[:label.shape[0]] = label
        padded_labels.append(padded_label)
    
    padded_labels = torch.stack(padded_labels)
    
    return images, padded_labels

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for epoch in range(50):
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

def evaluate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move images and labels to the GPU
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss}')

def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Move images and labels to the GPU
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            # Process outputs to detect defects
            # This part depends on your model's output format
            # For simplicity, assume outputs are bounding box coordinates
            defects = []
            for output in outputs:
                # Convert output to bounding box coordinates
                # Assuming output is in the format [x, y, w, h, confidence]
                if output[4] > 0.5:  # Confidence threshold
                    defects.append(output[:4].tolist())
            print("Detected Defects:", defects)

def main():
    # Set up dataset and data loader
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = WeldDefectDataset(r'C:\Users\vijay\OneDrive\Desktop\WDD\datasets\ir_images\train', transform=transform)
    val_dataset = WeldDefectDataset(r'C:\Users\vijay\OneDrive\Desktop\WDD\datasets\ir_images\val', transform=transform)
    test_dataset = WeldDefectDataset(r'C:\Users\vijay\OneDrive\Desktop\WDD\datasets\ir_images\test', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate)

    # Initialize model, device, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOModel()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, device, train_loader, optimizer, criterion)

    # Evaluate the model
    evaluate(model, device, val_loader, criterion)

    # Test the model
    test(model, device, test_loader)

if __name__ == '__main__':
    main()
