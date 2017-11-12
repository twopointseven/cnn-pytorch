import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

epochs = 1
batch_size = 1
learning_rate = 0.0001

train_dataset = dsets.MNIST(root='./data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False, 
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 10, kernel_size=3,padding=1), 
			nn.BatchNorm2d(10),
			nn.ReLU(),
			nn.MaxPool2d(2),
			)

		self.layer2 = nn.Sequential(
			nn.Conv2d(10, 10, kernel_size=3,padding=1),
			nn.BatchNorm2d(20),
			nn.ReLU(),
			)

		self.layer3 = nn.Sequential(
			nn.Conv2d(10, 20, kernel_size=3,padding=1),
			nn.BatchNorm2d(20),
			nn.ReLU(),
			)

		self.layer4 = nn.Sequential(
			nn.Conv2d(20, 20, kernel_size=3,padding=1),
			nn.BatchNorm2d(20),
			nn.ReLU(),
			nn.MaxPool2d(2)
			)


		self.layer5 = nn.Linear(7*7*20, 10) #28-14-7 (2 maxpools)
		self.layer6 = nn.Softmax2d()

	def forward(self, x):
		output1 = self.layer1(x)
		output2 = self.layer2(output1)
		output3 = self.layer3(output2)
		output4 = self.layer4(output3)
		output4 = output4.view(output3.size(0), -1)
		output5 = self.layer5(output4)
		output6 - self.layer6(output5)
		return output6

model = CNN()
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for j in range(epochs):
	for i, (images,labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)
		pred = model(images)
		optimizer.zero_grad()
		loss = criterion(pred,labels)
		loss.backward()
		optimizer.step()

		if (i % 1000 == 0):
			print("ITERATION: " + str(i))
			print("LOSS: " + str(loss))




print("TESTING")

c = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    OUT = model(images)
    _, pred = torch.max(OUT.data, 1)
    total += labels.size(0)
    c += (pred == labels).sum()

print(100 * c / total)
