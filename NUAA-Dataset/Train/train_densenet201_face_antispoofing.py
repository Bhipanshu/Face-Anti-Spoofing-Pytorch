import torch,os
import torch.nn as nn #helps us create and train neural networks
import torch.optim as optim #helps us with optimization
import torchvision #helps us with computer vision tasks
import torchvision.transforms as transforms #helps us transform our datasets
from torch.utils.data import DataLoader #helps us load our data
from torchvision import datasets #helps us download and import datasets
import torchvision.models as models

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_root = ''


batch_size = 32
train_dataset = datasets.ImageFolder(root = data_root,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

learning_rate = 0.001
num_classes = 2

model = models.densenet201(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


# print(model)

classifer = nn.Sequential(nn.Linear(model.classifier.in_features, 512),
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(512, num_classes))
model.classifier = classifer
# print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

modelSaveDir = os.path.join(os.getcwd(),'models_densenet201')
os.makedirs(modelSaveDir,exist_ok=True)
model.to(device)
model.train()

from thop import profile
input = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(input, ))
macs = macs/10**6
params = params/10**6
print(f"Total FLOPs: {macs:.2f} MFLOPs")
print(f'Total parameters: {params:.2f} Params(M)')

best_acc = 0.0

for epoch in range(100):
    running_loss = 0.0
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


print('Finished Training')