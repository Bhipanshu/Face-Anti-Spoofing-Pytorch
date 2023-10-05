import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from torch.utils.tensorboard import SummaryWriter

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


# Load a pretrained vgg16 model
vgg16_model = models.vgg16(pretrained=True)

# Modify the last fully connected layer for 10 classes  
num_features = vgg16_model.classifier[6].in_features
vgg16_model.classifier[6] = nn.Linear(num_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16_model.parameters(), lr=0.001)

modelSaveDir =  os.path.join(os.getcwd(), 'models_vgg16')
os.makedirs(modelSaveDir, exist_ok=True)

# Training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16_model.to(device)
vgg16_model.train()

## setup tensorboard
writer_train = SummaryWriter(log_dir=modelSaveDir+'/graph')

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

        outputs = vgg16_model(inputs)
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

    writer_train.add_scalar('Training Loss', (running_loss/len(train_loader)), epoch)
    writer_train.add_scalar('Training Accuracy',accuracy, epoch)

    if epoch == 0:
        best_acc = accuracy

    # Save model checkpoint 
    checkpoint = {'epoch': epoch + 1,'state_dict': vgg16_model.state_dict(),'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(modelSaveDir , f'checkpoint_fas_replayattack_vgg16.pth'))
    
    if accuracy > best_acc:

        torch.save(checkpoint, os.path.join(modelSaveDir , f'checkpoint_fas_replayattack_vgg16_acc_{accuracy:.4f}.pth'))

        best_acc=accuracy
print("Training completed!")

























#
