import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

from fine_tuning.algorithms.vae.utils import fetch_dataset
from torch.utils.data import TensorDataset, DataLoader

gpu = 6
n_epochs = 50
batch_size = 64

device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

X, y = fetch_dataset(root='/home/jdh/vae_training_data/',
                     classes=[0, 1, 2, 3, 0, 1, 2, 3],
                     folders=['noise', 'single_horizontal_with_compensation', 'single_vertical_with_compensation', 'triple_with_compensation',
                              'noise_2', 'single_horizontal_2', 'single_vertical_2', 'triple_with_compensation_2']
                     )
reshape_size = (224, 224)

# resize = transforms.Compose([
#     transforms.Resize(224),
#     # transforms.CenterCrop(224),
#     # transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

resize = transforms.Resize(size=reshape_size)

X = np.concatenate([X[..., np.newaxis]] * 3, axis=-1)
X = np.rollaxis(X, -1, 1)



tensor_x = torch.Tensor(X).to(device) # transform to torch tensor
tensor_y = torch.Tensor(y).type(torch.LongTensor).to(device)
tensor_x = resize(tensor_x)

train_size = int(0.8 * len(tensor_x))
test_size = 1 - train_size

train_dataset, test_dataset = torch.utils.data.random_split(list(zip(tensor_x, tensor_y)), [0.9, 0.1])

loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model = model.to(device)

# Freeze the weights of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


print('about to train')
# Train the model
for epoch in range(n_epochs):
    running_loss = 0.0
    total_correct = 0
    total_samples = 0


    for i, (inputs, labels) in tqdm(enumerate(loader, 0)):
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Update the running total of correct predictions and samples
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)


    # Print the average loss for the epoch
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(loader)))
    # Calculate the accuracy for this epoch
    accuracy = 100 * total_correct / total_samples
    print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x[None, ...].to(device=device)
            y = y[None, ...].to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    # model.train()


check_accuracy(test_dataset, model)
check_accuracy(train_dataset, model)