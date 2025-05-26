#exec.py
from net import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Executive:
	def __init__(self, train_batch=64, test_batch=100): #Maybe add lr and momentum for optimizor
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])
		train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
		test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

		self.train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
		self.test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)
		self.model = Net()
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
	
	def train(self, epochs=10, interval=100):
		for epoch in range(epochs):
			for i, (images, labels) in enumerate(self.train_loader):
				# Forward pass
				outputs = self.model(images)
				loss = self.criterion(outputs, labels)

				# Backward and optimize
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				if (i + 1) % interval == 0:
					print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
						  .format(epoch + 1, epochs, i + 1, len(self.train_loader), loss.item()))
	
	def load_model(self, file_name):
		self.model.load_state_dict(torch.load(file_name + ".pth"))
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		
	def save_model(self, file_name):
		torch.save(self.model.state_dict(), (file_name + ".pth"))
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
	
	def test(self):
		with torch.no_grad():
			correct = 0
			total = 0
			for images, labels in self.test_loader:
				outputs = self.model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

	def print(self, total=6, row=2, col=3):
		examples = enumerate(self.test_loader)
		batch_idx, (example_data, example_targets) = next(examples)
		with torch.no_grad():
			output = self.model(example_data)
		fig = plt.figure()
		for i in range(total):
			plt.subplot(row,col,i+1)
			plt.tight_layout()
			plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
			plt.title("Prediction: {}".format(
				output.data.max(1, keepdim=True)[1][i].item()))
			plt.xticks([])
			plt.yticks([])
		plt.show()
	def lint(self, total=6, row=2, col=3):
		examples = enumerate(self.test_loader)
		one, two, three = [], [], []
		for a, (b, c) in examples:
			one.append(a), two.append(b), three.append(c)
		with torch.no_grad():
			output = self.model(two[1])
		fig = plt.figure()
		for example_data in two:
			for i in range(total):
				plt.subplot(row,col,i+1)
				plt.tight_layout()
				plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
				plt.title("Prediction: {}".format(
					output.data.max(1, keepdim=True)[1][i].item()))
				plt.xticks([])
				plt.yticks([])
			plt.show()
			break
		
