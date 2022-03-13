import NN_model
import torch
from torch import nn
import numpy as np


def func(t, y):
    return 3 / 2 * y / (t + 1) + np.sqrt(t + 1)


def local_truncation_error(data, func):
    # R function
    # delta_x = data[1] - data[0]
    return 1/data[2]**2 * (data[4] - data[3] - data[2] * func(data[0], data[3]))


class TrainAndTest:
    def __init__(self, in_data, batch_size, device, lr=1e-3):
        self.model = NN_model.NeuralNetwork()
        self.in_data = in_data
        self.batch_size = batch_size
        self.loss_fn = nn.L1Loss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, func):
        # self.model.train()
        np.random.shuffle(self.in_data)  # blanda datamängd

        temp = torch.from_numpy(self.in_data).float()

        batch_truncation_error = torch.empty((1, self.batch_size))
        batch_pred = torch.empty((1, self.batch_size))
        # print(np.zeros((1, self.batch_size)))
        for index, data in enumerate(temp):

            # Compute prediction- and truncation- error
            # print(type(torch.narrow(data, 0, 0, 3)))
            batch_pred[0, index % self.batch_size] = self.model(data[:4])
            # print("here")
            batch_truncation_error[0, index % self.batch_size] = local_truncation_error(data, func)

            if index > 0 and (index+1) % self.batch_size == 0:

                # batch_pred = torch.from_numpy(batch_pred).float()
                # batch_truncation_error = torch.from_numpy(batch_truncation_error).float()
                # print(batch_pred, type(batch_pred))
                # print(batch_truncation_error, type(batch_truncation_error))


                # print(batch_truncation_error.size)
                # print(batch_pred.size)
                loss = self.loss_fn(batch_pred, batch_truncation_error).float()
                # loss = torch.autograd.Variable(loss, requires_grad=True)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                # print(self.model.linear_result_stack[0].weight.reta)
                self.optimizer.step()

                batch_truncation_error = torch.empty((1, self.batch_size))
                batch_pred = torch.empty((1, self.batch_size))

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
            # print(list(self.model.parameters()))


def main():
    in_data = np.load("outfile_exempel.npy")
    in_data = in_data[0]
    print(in_data)
    batch_size = 500
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")
    train = TrainAndTest(in_data=in_data,
                         batch_size=batch_size, device=device, lr=1e-5)
    train.model.train()
    for i in range(50):
        print("____________________")
        print("epoch:{}".format(i+1))
        print("____________________")
        train.train(func)


if __name__ == '__main__':
    main()
