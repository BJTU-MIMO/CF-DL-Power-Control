
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hdf5storage
import scipy.io as sio
import torch.utils.data as Data
import os
from matplotlib import pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("GPU型号： ", torch.cuda.get_device_name(0))

HDF5_DISABLE_VERSION_CHECK = 1



mat_contents = hdf5storage.loadmat('CSI_CF_training.mat')
mat_predict = hdf5storage.loadmat('CSI_CF_predict_8UE.mat')
Input = mat_contents['sumVal']
Input = np.transpose(Input)
Input = torch.FloatTensor(Input)

K = torch.FloatTensor(mat_contents['K'])
size = len(Input)

signal = mat_contents['signal_cell']
interference = mat_contents['interference_cell']
interference = np.transpose(interference)
G_cell = mat_contents['G_cell']
signal = torch.FloatTensor(signal)
interference = torch.FloatTensor(interference)
G_cell = torch.FloatTensor(G_cell)

# define hyperparameters
alpha = 2
amplitude = 2
bb = 10
Epoch = 300
Pmax = 100

BATCH_SIZE = 100

loader = Data.DataLoader(
    dataset=Input,
    batch_size=BATCH_SIZE,
    num_workers=0,
)


def calc_prod(SINR, bb):
    sum_loss = - torch.sum(bb * torch.log2(SINR))

    return sum_loss


def calc_maxmin(SINR, alpha):
    loss = - torch.mul(alpha, min(SINR))
    sum_loss = torch.sum(loss)
    return sum_loss


def calc_SINR(power, signal, interference, index):
    denominator = torch.mv(torch.Tensor(interference[index, :, :]), power) + 1
    numerator = torch.mul(signal[:, index], power)
    SINR = numerator / denominator
    return SINR



def backgrad(paras):
    for name, weight in paras:

        if weight.requires_grad:

            print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = 100 * torch.sigmoid(self.output(x))
        return x


if __name__ == '__main__':
    model = ANN().cuda()


    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    output = []
    lr = 0.3

    # define variable for loss compare
    loss_p = []

    for epoch in range(Epoch):
        # change learning rate
        if epoch > 0 and epoch % 200 == 0:
            lr /= 3
            adjust_learning_rate(optimizer, lr)
            print('change learning_rate to {}'.format(lr))

        for step, batch_x in enumerate(loader):
            batch_x = batch_x.cuda()
            predict_power = model.forward(batch_x).cpu()

            loss_func_maxmin = 0
            loss_func_prod = 0
            length = len(batch_x)
            for i in range(length):
                index = step * BATCH_SIZE + i
                SINR = calc_SINR(predict_power[i], signal, interference, index)
                loss_func_maxmin += calc_maxmin(SINR, alpha) / length

            optimizer.zero_grad()
            loss_func_maxmin.backward()

            optimizer.step()
            # training set save
            if epoch == Epoch - 1:
                for nn in range(length):
                    output.append(predict_power[nn].detach().numpy())



        print(f'Epoch: {epoch}, Loss: {"{:.4f}".format(loss_func_maxmin)}')
        loss_p.append(loss_func_maxmin.detach().numpy())


    # predict the test set
    index = -1
    Input_test = mat_predict['sumVal_pr']
    Input_test = np.transpose(Input_test)
    Input_test = torch.FloatTensor(Input_test)
    size_testset = len(Input_test)
    output_test = []

    while index < size_testset - 1:
        index += 1
        predict_power_test = model.forward(Input_test[index, :].cuda()).cpu()
        output_test.append(predict_power_test.detach().numpy())

    # save the training_set and predict_set
    sio.savemat('output-minibatch-newloss-others.mat',{'pBest_compare': np.transpose(output), 'pBest_test_compare': np.transpose(output_test),
                 'loss_me_compare': loss_p})

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss of Max-min', fontsize=20)
    plt.plot(np.arange(0, Epoch), loss_p)
    plt.savefig('./Loss of Max-min-others.jpg')
    plt.show()