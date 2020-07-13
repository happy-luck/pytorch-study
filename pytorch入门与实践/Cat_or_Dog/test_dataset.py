from data.dataset import DogCat
from torch.utils.data import DataLoader

train_dataset = DogCat('data/train/', train=True)
trainloader = DataLoader(train_dataset,
                        batch_size = 128,
                        shuffle = True,
                        num_workers = 4)
                  
for ii, (data, label) in enumerate(trainloader):
	print(data.size())
print('finish')