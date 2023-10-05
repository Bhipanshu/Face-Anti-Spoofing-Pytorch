import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 #Define the path to the root folder containing your class folders
data_root = ''

# Create a dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=data_root, transform=transform)

# Create a DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


learning_rate = 0.001
num_classes = 2


# Load a pretrained ResNet18 model
resnet18_model = models.resnet18(pretrained=True)

# Modify the last fully connected layer for 10 classes

in_features = resnet18_model.fc.in_features
resnet18_model.fc = nn.Linear(in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18_model.parameters(), lr=0.001)

modelSaveDir =  os.path.join(os.getcwd(), 'models_resnet18')
os.makedirs(modelSaveDir, exist_ok=True)

# Training loop

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
resnet18_model.to(device)
resnet18_model.train()

best_acc = 0

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet18_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], iter[{i}/{len(train_loader)}] Loss: {loss.item():.4f} Accuracy: {accuracy:.4f}%', end='\r')

    print(f'Epoch [{epoch+1}/{num_epochs}], iter[{i}/{len(train_loader)}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {accuracy:.4f}%')

    if epoch == 0:
        best_acc = accuracy

    checkpoint = {'epoch': epoch + 1,'state_dict': resnet18_model.state_dict(),'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(modelSaveDir , f'checkpoint_fas_resnet18.pth'))
    
    if accuracy > best_acc:

        torch.save(checkpoint, os.path.join(modelSaveDir , f'checkpoint_fas_resnet18_acc_{accuracy:.4f}.pth'))

        best_acc=accuracy
print("Training completed!")

























#