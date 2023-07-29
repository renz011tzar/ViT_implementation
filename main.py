import torch
import torchvision
import torchvision.transforms as transforms
from torch import utils
import torch.optim as optim
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from DP_implementation.model.network import *
from DP_implementation.training.train import *
from DP_implementation.functions.load_data import *
from DP_implementation.functions.load_data import create_transform
from DP_implementation.functions.loss import *
from DP_implementation.functions.optimizer import *

batch_size=64
transform=create_transform()
trainpath= './data/train'
testpath= './data/test'

train_set = load_data(trainpath, transform)
test_set = load_data(testpath, transform)

train_loader=create_data_loader(data=train_set, batch_size=batch_size, shuffle=True, num_workers= 2)
test_loader=create_data_loader(data=test_set, batch_size=batch_size, shuffle=True, num_workers= 2)

classes=get_classes(test_set)
device = "cuda" if torch.cuda.is_available() else "cpu"
net=ViT()
net = net.to(device)

loss_fn=get_criterion()
optimizer=get_optimizer_2(net, 1e-3, 0.3)
lambda1 = lambda epoch: 1 - epoch / 100  # Assuming you have 100 epochs
scheduler = CosineAnnealingLR(optimizer, T_max=50)

outcome= train(
    model=net,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    device=device,
    checkpoint_dir="modelos"
)
