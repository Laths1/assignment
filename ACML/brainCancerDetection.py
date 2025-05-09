import numpy as numpy
from PIL import Image
import torch
import random
import torch.nn as nn
import torch.nn.functional as f
from torchvision.transforms import functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class DataHandler:

    @staticmethod
    def transform(augment=False):
        base_transforms = [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]

        if augment:
            augmentation_transforms = [
                transforms.RandomVerticalFlip(),
                transforms.Lambda(lambda x: F.rotate(x, random.choice([90, 180, 270])))
            ]
            return transforms.Compose(augmentation_transforms + base_transforms)  
        
        return transforms.Compose(base_transforms)

    @staticmethod
    def dataSplit(originalDataset, augmentedDataset):
        allData = torch.utils.data.ConcatDataset([originalDataset, augmentedDataset])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return kf.split(allData)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  #(12, 503, 503 )
        self.pool1 = nn.MaxPool2d(2, 2)  #(12, 251, 251)
        self.conv2 = nn.Conv2d(6, 16, 5)  #(24, 242, 242) -> (24, 121, 121) 
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.fc1 = None 
        self._fc1_init = False 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        

    def forward(self, input):
        x = self.pool1(f.relu(self.conv1(input)))
        x = self.pool2(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 

        if not self._fc1_init:
            self.fc1 = nn.Linear(x.shape[1], 120).to(x.device)
            self._fc1_init = True

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x

class ModelTest:

    @staticmethod
    def plotLoss(lossList):
        plt.plot(lossList, label='Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def cofusionMatrix(actual, predicted):
        cm = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
        cm_display.plot()
        plt.show()

if __name__ == "__main__":
    data_dir = r"C:\Users\fzm1209\Downloads\archive\Brain_Cancer raw MRI data\Brain_Cancer"
    originalData = torchvision.datasets.ImageFolder(root=data_dir, transform=DataHandler.transform())
    augmentedData = torchvision.datasets.ImageFolder(root=data_dir, transform=DataHandler.transform(augment=True))

    allData = torch.utils.data.ConcatDataset([originalData, augmentedData])
    
    # Implement batch processing
    batch_size = 16
    train_loader = DataLoader(allData, batch_size=batch_size, shuffle=True)

    # Initialize model
    net = Model()

    # Train model in batches
    numOfEpochs = 10
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(numOfEpochs):
        totalLoss = []
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            lossValue = loss(outputs, labels)
            lossValue.backward()
            optimizer.step()
            totalLoss.append(lossValue.item())

        print(f"Epoch {epoch+1}, Loss: {sum(totalLoss)/len(totalLoss)}")
