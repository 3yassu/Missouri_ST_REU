import install
import load
import madgrad
import point_trans
import rand
from config import Config
from point_trans import (PointTransformerBlock, PointTransformerClassif)
import matplotlib.pyplot as plt

class Executive():
	def __init__(self):
		self.val_accs = []
		self.train_accs = []
		self.device = point_trans.torch.device("cuda:0" if point_trans.torch.cuda.is_available() else "cpu")
		self.device
		data = load.ModelNetDataLoader('./sample_data/modelnet40_normal_resampled/',
                          split='test',
                          process_data=True)
		DataLoader = point_trans.torch.utils.data.DataLoader(data, batch_size=4096, shuffle=True)
		for point, label in DataLoader:
			print(point.shape)
			print(label.shape)
			

		self.cfg = Config(
			model = Config(nneighbor = 16, nblocks = 4, transformer_dim = 64),
			batch_size = 64,
			epoch = 200,
			learning_rate = 5e-3,
			gpu = 1,
			num_point = 1024,
			optimizer = 'SGD',
			weight_decay = 1e-4,
			normal = True
		)
		self.cfg.num_class = 40
		self.cfg.input_dim = 6 if self.cfg.normal else 3

		self.model = PointTransformerClassif(self.cfg).to(self.device)

		self.train_loader = point_trans.torch.utils.data.DataLoader(
			load.ModelNetDataLoader('./sample_data/modelnet40_normal_resampled/', split='train', process_data=True, transforms=None),
			batch_size=self.cfg.batch_size,
			shuffle=True
		)

		self.test_loader = point_trans.torch.utils.data.DataLoader(
			load.ModelNetDataLoader('./sample_data/modelnet40_normal_resampled/', split='test', process_data=True, transforms=None),
			batch_size=64,
			shuffle=True
		)
	def print_model(self):
		print(self.model)
	def train(self, epochs=100, val_step=5):
		if epochs is None:
			epochs = self.cfg.epoch

		criterion = point_trans.nn.CrossEntropyLoss()
		if self.cfg.optimizer == 'Adam':
			optimizer = point_trans.torch.optim.Adam(
				self.model.parameters(),
				lr=self.cfg.learning_rate,
				betas=(0.9, 0.999),
				eps=1e-08,
				weight_decay=self.cfg.weight_decay
			)
		elif self.cfg.optimizer == 'MadGrad':
			optimizer = madgrad.MADGRAD(model.parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay)
		else:
			optimizer = point_trans.torch.optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay)
		scheduler = point_trans.torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs*6//10,epochs*8//10], gamma=0.1)

		self.val_accs = []
		self.train_accs = []
		best_val_acc = -1.0
		loss=0
		for epoch in load.tqdm(range(epochs), position=0, leave=True):
			self.model.train()
			correct = total = 0
			for i, data in enumerate(self.train_loader, 0):
				inputs, labels = data
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				optimizer.zero_grad()
				outputs = self.model(inputs)
				loss = criterion(outputs, labels.long())
				loss.backward()
				optimizer.step()
				_, predicted = point_trans.torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			train_acc = 100. * correct / total
			self.train_accs.append(train_acc)

			if (epoch+1) % val_step  == 0:
				model.eval()
				correct = total = 0
				if self.test_loader:
					with point_trans.torch.no_grad():
						for data in self.test_loader:
							inputs, labels = data
							inputs, labels = inputs.to(self.device), labels.to(self.device)
							outputs = self.model(inputs)
							_, predicted = point_trans.torch.max(outputs.data, 1)
							total += labels.size(0)
							correct += (predicted == labels).sum().item()
					val_acc = 100. * correct / total
					self.val_accs.append(val_acc)
					print('\n Epoch: %d, Train accuracy: %.1f %%, Test accuracy: %.1f %%' %(epoch+1, train_acc, val_acc))
				if self.val_accs[-1] > best_val_acc:
					point_trans.torch.save(model.state_dict(), 'checkpoint.pth')
			else:
				print('\n Epoch: %d, Train accuracy: %.1f %%' %(epoch+1, train_acc))

			scheduler.step()
		
	def save(self):
		point_trans.torch.save(model.state_dict(), 'checkpoint_last.pth')
			
	def print(self):		
		plt.plot(self.train_accs, label="train")
		plt.show()
		plt.plot([10*i for i in range(1,(len(self.val_accs)+1))],self.val_accs,label="val")
		plt.show()
			
	def dumpy():
		pickle.dump([self.train_accs, self.val_accs], open('accs_100epoch_transdim128.p', 'wb'))
