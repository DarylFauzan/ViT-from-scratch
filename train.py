import torch, numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from model import *
from trainer import *

# define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# create composer for image transformation 
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(280),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

# # Training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# # Number of images per class you want (if want to train only subset of the train data)
# n_per_class = 3000

# # Build indices for the subset
# targets = np.array(trainset.targets)
# subset_indices = []

# for class_id in range(len(trainset.classes)):  
#     class_indices = np.where(targets == class_id)[0]
#     chosen_indices = np.random.choice(class_indices, n_per_class, replace=False)
#     subset_indices.extend(chosen_indices)

# # Create subset
# trainset = Subset(trainset, subset_indices)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True)


# Initiate Model
model_config = ModelConfig()
model = VisualTransformerClassifier(32, 8, model_config).to(device)

# initiate trainer
train_config = TrainConfig()
trainer = ClassificationTrainer(model, train_config, trainset, testset, task = 'multiclass', device = device)

# train the model
trainer.train()

# save the model
torch.save(model.state_dict(), "model/Jero_ViT_CIFAR10.pth") 