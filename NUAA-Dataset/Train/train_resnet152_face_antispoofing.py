import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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

model = torchvision.models.resnet152()
print(model)
# classifier = nn.Linear(1000, num_classes)
in_features = model.fc.in_features

model.fc = nn.Linear(in_features, num_classes)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

modelSaveDir =  os.path.join(os.getcwd(), 'models_resnet152')
os.makedirs(modelSaveDir, exist_ok=True)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

best_acc = 0

num_epochs = 100


from thop import profile
input = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(input, ))
macs = macs/10**6
params = params/10**6
print(f"Total FLOPs: {macs:.2f} MFLOPs")
print(f'Total parameters: {params:.2f} Params(M)')

for epoch in range(num_epochs):
    running_loss = 0
    correct = 0
    total = 0

    for i , data in enumerate(train_loader,0):
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # print('Epoch: %d | Loss: %.4f | Accuracy: %.4f'%(epoch+1,running_loss,accuracy))

        print(f'Epoch [{epoch+1}/{100}], iter[{i}/{len(train_loader)}] Loss: {loss.item():.4f} Accuracy: {accuracy:.4f}%', end='\r')

    if epoch == 0:
        best_acc = accuracy

    checkpoint = {'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}

    torch.save(checkpoint, os.path.join(modelSaveDir, 'epoch_%d.pth' % (epoch + 1)))

    if accuracy > best_acc:
        torch.save(checkpoint, os.path.join(modelSaveDir, 'best.pth'))

        best_acc = accuracy
    print(f'Epoch [{epoch+1}/{100}], iter[{i}/{len(train_loader)}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {accuracy:.4f}%')



    
