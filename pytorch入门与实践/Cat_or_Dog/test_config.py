import models
from config import DefaultConfig
from data.dataset import  DogCat
opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root)