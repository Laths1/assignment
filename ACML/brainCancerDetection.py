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
class DataHandler:

  @staticmethod
  def transform(augment=False):
    base_transforms = [
      transforms.Resize((512, 512)),
      transforms.ToTensor(),
      transforms.Normalize([0.5],[0.5])
    ]
    if augment:
      augmentation_transforms = [
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: F.rotate(x, random.choice([90, 180, 270]))),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
        ]
      return transforms.Compose(augmentation_transforms + base_transforms)
    return transforms.Compose(base_transforms)

  @staticmethod
  def dataSplit(originalDataset, augmentedDataset):
    # 5 fold cross validation
    allData = torch.utils.data.ConcatDataset([originalDataset, augmentedDataset])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return kf.split(allData)
class Model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,6,5) #(12, 503, 503 )
    self.pool1 = nn.MaxPool2d(2,2) #(12, 251, 251)
    self.conv2 = nn.Conv2d(6,16,5) #(24, 242, 242) -> (24, 121, 121)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 3)

  def forward(self, input):
    x = self.pool1(f.relu(self.conv1(input)))
    x = self.pool1(f.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
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
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
    cm_display.plot()
    plt.show()

if __name__ == "__main__":
    data_dir = r"C:\Users\lathi\Documents\SCHOOL\Honours\AI\assignment\ACML\archive\Brain_Cancer_raw_MRI_data\Brain_Cancer"
    originalData = torchvision.datasets.ImageFolder(root=data_dir, transform=DataHandler.transform())
    augmentedData = torchvision.datasets.ImageFolder(root=data_dir, transform=DataHandler.transform(augment=True))

    # allData = DataHandler.dataSplit(originalData, augmentedData)
    allData = torch.utils.data.ConcatDataset([originalData, augmentedData])
    # model
    net = Model()

    # train
    numOfEpochs = 10
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    for epoch in range(numOfEpochs):
       totalLoss = []
       for i, data in enumerate(allData, 0):
           inputs, labels = data
           optimizer.zero_grad()
           outputs = net(inputs)
           lossValue = loss(outputs, labels)
           lossValue.backward()
           optimizer.step()
           print(lossValue.item())
           totalLoss.append(lossValue.item())

    

    
 

  
