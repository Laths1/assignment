import numpy as np
from PIL import Image
import torch
import random
import torch.nn as nn
import torch.nn.functional as f
from torchvision.transforms import functional as F
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
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
        transforms.Lambda(lambda x: F.rotate(x, random.choice([90, 180, 270])))
        ]
      return transforms.Compose(augmentation_transforms + base_transforms)
    return transforms.Compose(base_transforms)

  @staticmethod
  def dataSplit(trainSet):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(len(trainSet))
    train_subsets = []
    val_subsets = []
    for train_idx, val_idx in kf.split(indices):
        train_subsets.append(torch.utils.data.Subset(trainSet, train_idx))
        val_subsets.append(torch.utils.data.Subset(trainSet, val_idx))
    
    return train_subsets, val_subsets  # Returns two lists of dataset subsets
class Model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)  # (6, 508, 508)
    self.pool1 = nn.MaxPool2d(2, 2)   # (6, 254, 254)
    self.conv2 = nn.Conv2d(6, 16, 5) # (16, 250, 250)
    self.fc1 = nn.Linear(16*125*125, 120)  
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
  def plotLoss(lossList, label):
    plt.plot(lossList, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Trainig loss')
    plt.title('Performance')
    plt.legend()
    plt.show()

  @staticmethod
  def plotLoss(lossList, valList):
    plt.plot(lossList, label='Loss')
    plt.plot(valList, label='Validation')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Performance')
    plt.legend()
    plt.show()

  @staticmethod
  def cofusionMatrix(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
    cm_display.plot()
    plt.show()

if __name__ == "__main__":
    data_dir = r"C:\Users\lathi\Documents\SCHOOL\Honours\AI\assignment\ACML\archive\Brain_Cancer raw MRI data\Brain_Cancer"
    originalData = torchvision.datasets.ImageFolder(root=data_dir, transform=DataHandler.transform())
    augmentedData = torchvision.datasets.ImageFolder(root=data_dir, transform=DataHandler.transform(augment=True))

    # data
    allData = torch.utils.data.ConcatDataset([originalData, augmentedData])
    train_ratio = 0.8
    test_ratio = 0.2
    train_size = int(train_ratio * len(allData))
    test_size = len(allData) - train_size
   
    train_set, test_set = random_split(allData, [train_size, test_size])
    train_batches, val_sets = DataHandler.dataSplit(train_set)

    group = 0

    trainLoader = torch.utils.data.DataLoader(train_batches[group], batch_size=32, shuffle=True)
    validationLoader = torch.utils.data.DataLoader(val_sets[group], batch_size=32, shuffle=True)

    #model
    net = Model()

    # train
    numOfEpochs = 10
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 
    totalLoss = []
    validationLoss = []
    accuracy = []

    for epoch in range(numOfEpochs):
      runningLoss = 0.0

      #trainig phase
      for i, data in enumerate(trainLoader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        lossValue = loss(outputs, labels)
        lossValue.backward()
        optimizer.step()
        runningLoss += lossValue.item()
      totalLoss.append(runningLoss / len(trainLoader))
      print(f"Epoch {epoch+1} Loss: {runningLoss/len(trainLoader)}")

      #validation phase
      net.eval()
      correct = 0
      total = 0
      with torch.no_grad():
        runningLoss = 0.0
        for i, data in enumerate(validationLoader):
          inputs, labels = data
          outputs = net(inputs)
          lossValue = loss(outputs, labels)
          runningLoss += lossValue.item()
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      validationLoss.append(runningLoss / len(validationLoader))
      accuracy.append(100 * correct / total)
      print(f"Validation Loss: {runningLoss/len(validationLoader)}")

    torch.save(net.state_dict(), 'brain_cancer_model_weights.pth')

    #model evaluation
    ModelTest.plotLoss(totalLoss, validationLoss)
    ModelTest.plotLoss(accuracy, 'Accuracy')

    # model test
    

  

    

    
 

  
