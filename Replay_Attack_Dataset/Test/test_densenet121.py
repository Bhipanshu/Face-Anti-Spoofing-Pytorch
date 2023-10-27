
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets


# Define data transformations (should match the ones used for training)
transform = transforms.Compose([
	transforms.Resize((224, 224)),  # Resize to match CIFAR-10 size
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Assuming you used this normalization for training
])

data_root = 'path to dataset'

train_dataset = datasets.ImageFolder(root=data_root, transform=transform)

len(train_dataset)
test_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

# Load a pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the last fully connected layer for 10 classes in CIFAR-10
num_classes = 2
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

dict_net = torch.load('saved model')
model.load_state_dict(dict_net["state_dict"])



# Set the model to evaluation mode
model.eval()

# Define a device for running the model (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize variables for calculating accuracy
correct = 0
total = 0

def add_gaussian_noise(image, mean=0, std=0.3):
    noise = torch.randn_like(image) * std + mean
    noisy_image = torch.clamp(image + noise, min=0, max=1)  # Clip values to [0, 1] range
    return noisy_image

# # #function for random noise
def add_noise(image, noise_level):
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)

for i in range(1, 10):
	# Don't compute gradients during testing
	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)

			n = i *0.01

			noise_level =i*n
			# images = add_noise(images,noise_level)


			# images = add_gaussian_noise(images, mean=n, std=0)

			# Forward pass
			outputs = model(images)
			
			# Get predicted labels
			_, predicted = torch.max(outputs.data, 1)
			
			# Update total and correct predictions
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			accuracy = 100 * correct / total
			# print(f'Test Accuracy for std :{n} :{accuracy:.4f}%',end='\r')
			# print(f'Test Accuracy for std :{n} :{accuracy:.4f}%',end='\r')

			print(f'Test Accuracy for mean :{n} :{accuracy:.4f}%',end='\r')
			# print(f'Test Accuracy for noise_level :{n} :{accuracy:.4f}%',end='\r')

	# Calculate accuracy
	accuracy = 100 * correct / total

	# print(f'Test Accuracy: {accuracy:.2f}%')
	# print(f'Test Accuracy for std {n}: {accuracy:.4f}%')
	print(f'Test Accuracy for mean {n}: {accuracy:.4f}%')
	# print(f'Test Accuracy for noise_level{n}: {accuracy:.4f}%')


