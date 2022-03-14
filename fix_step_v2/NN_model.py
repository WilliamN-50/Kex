import torch
from torch import nn
import numpy as np
import GenerateData as gd

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self, num_y):
        super(NeuralNetwork, self).__init__()  # Take the init from nn.Module
        self.linear_result_stack = nn.Sequential(
            nn.Linear(3+num_y, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, num_y)
        )

    def forward(self, x):  # Ett steg beräkning, x = [xi, xi+1, yi]
        x = self.linear_result_stack(x)
        return x
        # return result.detach().numpy()


class TrainAndTest:
    def __init__(self, diff_eq, in_data, batch_size, device, train_ratio=0.85, lr=1e-3):
        self.model = NeuralNetwork(diff_eq.num_y)
        self.diff_eq = diff_eq
        self.train_data, self.test_data = self.split_train_test(in_data, train_ratio)
        self.batch_size = batch_size
        self.loss_fn = nn.L1Loss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    @ staticmethod
    def split_train_test(in_data, ratio):
        np.random.shuffle(in_data)
        num_train_data = int(in_data.shape[0]*ratio)
        train_data = in_data[:num_train_data]
        test_data = in_data[num_train_data:]
        return train_data, test_data

    def train(self):
        self.model.train()
        np.random.shuffle(self.train_data)

        torch_data = torch.from_numpy(self.train_data).float()

        batch_truncation_error = torch.empty((self.batch_size, self.diff_eq.num_y))
        batch_pred = torch.empty((self.batch_size, self.diff_eq.num_y))

        for index, data in enumerate(torch_data):

            # Compute prediction- and truncation- error
            batch_pred[index % self.batch_size, :] = self.model(data[: 3 + self.diff_eq.num_y])
            # print(local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y))
            batch_truncation_error[index % self.batch_size, :] = local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y)

            if index > 0 and (index+1) % self.batch_size == 0:
                loss = 0
                for i in range(self.diff_eq.num_y):
                    loss = loss + self.loss_fn(batch_pred[:, i], batch_truncation_error[:, i]).float()

                loss = loss / self.diff_eq.num_y
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_truncation_error = torch.empty((self.batch_size, self.diff_eq.num_y))
                batch_pred = torch.empty((self.batch_size, self.diff_eq.num_y))

                loss, current = loss.item(), index+1
                print(f"loss:{loss:>7f} [{current:>5d}/{self.train_data.shape[0]:>5d}]")

        rest = self.train_data.shape[0] % self.batch_size
        if rest != 0:
            loss = self.loss_fn(batch_pred[:rest], batch_truncation_error[:rest])

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), self.train_data.shape[0]
            print(f"loss:{loss:>7f} [{current:>5d}/{self.train_data.shape[0]:>5d}]")
            # print(list(self.model.parameters()))

    def test(self):
        self.model.eval()
        torch_data = torch.from_numpy(self.test_data).float()
        with torch.no_grad():

            batch_truncation_error = torch.empty((self.test_data.shape[0], self.diff_eq.num_y))
            batch_pred = torch.empty((self.test_data.shape[0], self.diff_eq.num_y))

            for index, data in enumerate(torch_data):
                # Compute prediction- and truncation- error
                # print(data[: 3 + self.diff_eq.num_y])
                batch_pred[index, :] = self.model(data[: 3 + self.diff_eq.num_y])
                # print("here")
                # print(local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y))
                batch_truncation_error[index, :] = local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y)

            test_loss = 0
            for i in range(self.diff_eq.num_y):
                test_loss = test_loss + self.loss_fn(batch_pred[:, i], batch_truncation_error[:, i]).float()

            test_loss = test_loss / self.diff_eq.num_y
            # print("here")

            test_loss, current = test_loss.item(), self.test_data.shape[0]
            print(f"test_loss:{test_loss:>7f} [{current:>5d}/{self.test_data.shape[0]:>5d}]")


def local_truncation_error(data, func, num_y):
    # R function
    y_first = data[3:3+num_y]
    y_second = data[3+num_y:]
    # print(torch.matmul(data[2], func(data[0], y_first)))
    # print(func(data[0], y_first))
    # print(1/data[2]**2 * (y_second - y_first - data[2] * func(data[0], y_first)))
    return 1/data[2]**2 * (y_second - y_first - data[2] * func(data[0], y_first))


def main():
    in_data = np.load("outfile_exempel.npy")
    batch_size = 500
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")
    train = TrainAndTest(diff_eq=gd.Diff_eq_1(t_0=0, t_end=10, y_0=[1, 2]), in_data=in_data,
                         batch_size=batch_size, device=device, lr=1e-3)
    for i in range(50):
        print("____________________")
        print("epoch:{}".format(i + 1))
        print("____________________")
        train.train()
        train.test()


if __name__ == '__main__':
    main()
