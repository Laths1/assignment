import numpy as np
from PIL import Image
import torch
import random
import queue
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
import multiprocessing

class DataHandler:

  @staticmethod
  def rotate_image(x):
    return F.rotate(x, random.choice([90, 180, 270]))

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
        transforms.Lambda(DataHandler.rotate_image)
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
  def plotAccuracy(lossList, label, id):
    plt.plot(lossList, label=label)
    plt.xlabel('Episodes')
    plt.ylabel('Trainig loss')
    plt.title(f'Performance {id}')
    plt.legend()
    plt.show()

  @staticmethod
  def plotLoss(lossList, valList,id):
    plt.plot(lossList, label='Loss')
    plt.plot(valList, label='Validation')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title(f'Performance {id}')
    plt.legend()
    plt.show()

  @staticmethod
  def cofusionMatrix(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
    cm_display.plot()
    plt.show()

class Process(multiprocessing.Process):

    def __init__(self, train, val, batch, lr, param, id, result_queue, epochs):
        super(Process,self).__init__()
        self.id = id
        self.train = train
        self.val = val
        self.batch = batch
        self.lr = lr
        self.param = param
        self.result_queue = result_queue
        self.epochs = epochs

    def run(self):
        trainLoader = torch.utils.data.DataLoader(self.train, batch_size=32, shuffle=True)
        validationLoader = torch.utils.data.DataLoader(self.val, batch_size=32, shuffle=True)

        #model
        net = Model()

        # train
        numOfEpochs = self.epochs
        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9) 
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
            # print(f"Process {self.id} Epoch {epoch+1} Loss: {runningLoss/len(trainLoader)}")

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
            # print(f"Process {self.id} Validation Loss: {runningLoss/len(validationLoader)}")

        torch.save(net.state_dict(), self.param)
        self.result_queue.put({
            'id': self.id,
            'train_loss': totalLoss,
            'val_loss': validationLoss,
            'accuracy': accuracy
        })

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

    # parallel training
    result_q = multiprocessing.Queue()
    results = []

    processes = [
        Process(train_batches[0], val_sets[0], 0, 0.001, 'model1.pth', 1, result_q, 5),   # Low LR, few epochs
        Process(train_batches[1], val_sets[1], 1, 0.01, 'model2.pth', 2, result_q, 10),   # Default
        Process(train_batches[2], val_sets[2], 2, 0.02, 'model3.pth', 3, result_q, 15),   # Higher LR, more epochs
        Process(train_batches[3], val_sets[3], 3, 0.005, 'model4.pth', 4, result_q, 20),  # Very low LR, many epochs
        Process(train_batches[4], val_sets[4], 4, 0.01, 'model5.pth', 5, result_q, 15)    # Default, higher epochs
    ]

    try:
        
        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=86400)  # 24-hour timeout

        results = []
        while len(results) < len(processes):
            try:
                results.append(result_q.get(timeout=14400))
            except queue.Empty:
                print("Timeout waiting for results!")
                break

        # Plot results
        for res in sorted(results, key=lambda x: x['id']):  # Plot in order
            ModelTest.plotLoss(res['train_loss'], res['val_loss'], res['id'])
            ModelTest.plotAccuracy(res['accuracy'], res['id'])

    except Exception as e:
        print(f"Error: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
    finally:
        for p in processes:
            p.join() 
    # model test
    

  

    

    
 

  
