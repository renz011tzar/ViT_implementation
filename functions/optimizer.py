import torch.optim as optim

def get_optimizer_1(net, lr, momentum):
    return optim.SGD(net.parameters(), lr=lr, momentum=momentum)

def get_optimizer_2(net, lr, weight_decay):
    return optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
