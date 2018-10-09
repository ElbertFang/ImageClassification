import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.autograd import Variable

num_epochs = 10
batch_size = 64
learning_rate = 0.001
num_workers = 3
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
use_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])
#Cifar-10 dataset
train_data = datasets.CIFAR10(root = './dataset', train = True, download = True, transform = use_transforms)
test_data = datasets.CIFAR10(root = './dataset', train = False, download = True, transform = use_transforms)

#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True,
                                         num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle = True,
                                         num_workers = num_workers)

#CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cnn_2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # self.cnn_3 = nn.Sequential(
        #     nn.Conv2d(96, 1, kernel_size = 1, padding = 0),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(),
        # )
        self.cnn_3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size = 5, padding = 0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        #self.fc = nn.Linear(8*8, 10)
        self.fc = nn.Sequential(
            nn.Linear(4*4*192, 10),
            #nn.ReLU(),
            #nn.Dropout(0.3)
        )

    def forward(self, input):
        output = self.cnn_1(input)
        output = self.cnn_2(output)
        output = self.cnn_3(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

#Model, loss and optimizer
model = CNN().cuda()
XE_loss = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Train step
model.train()
for epoch in range(num_epochs):
    for index, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = XE_loss(output, labels)
        loss.backward()
        optimizer.step()

        if((index+1)%200 == 0):
            print("Epoch[%d/%d] Iter[%d/%d] XE_loss : %.4f"
                  %(epoch+1, num_epochs, index+1, len(train_data)//batch_size, loss.data[0]))

#Test step
model.eval()
correct = 0
num = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    labels = labels.cuda()
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    num += labels.size(0)
    correct += (predicted == labels).sum()

print("The test result is %.2f"%(100*correct/num))

#Save the model
torch.save(model.state_dict(),'model.pkl')