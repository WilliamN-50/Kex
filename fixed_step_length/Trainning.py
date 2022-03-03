import NN_model
import torch
from torch import nn
from torchvision.transforms import ToTensor
import numpy as np


def local_truncation_error(data, func):
    # R function
    delta_x = data[1] - data[0]
    return 1/delta_x**2 * (data[3] - data[2] - delta_x * func(data[0], data[2]))


class TrainAndTest:
    def __init__(self, in_data, batch_size, device, lr=1e-3):
        self.model = NN_model.NeuralNetwork().to(device)
        self.in_data = in_data
        self.batch_size = batch_size
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, func):
        self.model.train()
        np.random.shuffle(self.in_data)  # blanda datamängd

        batch_truncation_error = np.zeros(self.batch_size, 1)
        batch_pred = np.zeros(self.batch_size, 1)
        for index, data in enumerate(self.in_data):
            data = data.to(self.device)

            # Compute prediction- and truncation- error
            batch_pred[index % self.batch_size] = self.model(data[:3])
            batch_truncation_error[index % self.batch_size] = local_truncation_error(data, func)

            if index > 0 and (index+1) % self.batch_size == 0:
                loss = self.loss_fn(batch_pred, batch_truncation_error)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_truncation_error = np.zeros(self.batch_size, 1)
                batch_pred = np.zeros(self.batch_size, 1)

                loss, current = loss.item(), index+1
                print(f"loss:{loss:>7f} [{current:>5d}/{len(self.in_data):>5d}]")

        rest = len(self.in_data) % self.batch_size
        if rest != 0:

            loss = self.loss_fn(batch_pred[:rest], batch_truncation_error[:rest])

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), len(self.in_data)
            print(f"loss:{loss:>7f} [{current:>5d}/{len(self.in_data):>5d}]")



