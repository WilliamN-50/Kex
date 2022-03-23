import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import GenerateData as gd

# Define model


class NeuralNetwork(nn.Module):
    """
    ____________________________
    The NeuralNetwork class.
    Constructs a NN model for predicting the local error of the Euler method.
    ____________________________
    """
    def __init__(self, num_y):
        super(NeuralNetwork, self).__init__()  # Take the init from nn.Module
        self.linear_result_stack = nn.Sequential(
            nn.Linear(2+num_y, 80),
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

    def forward(self, x):
        x = self.linear_result_stack(x)
        return x


class TrainAndTest:
    """
    ____________________________
    The TrainAndTest class.
    Trains and tests the NeuralNetwork model.
    ____________________________
    """
    def __init__(self, model, diff_eq, in_data, batch_size, device, train_ratio=0.85, lr=1e-3):
        self.model = model
        self.diff_eq = diff_eq
        self.train_data, self.test_data = self._split_train_test(in_data, train_ratio, device)
        self.batch_size = batch_size
        self.loss_fn = nn.L1Loss(reduction="mean")
        self.test_loss = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    @ staticmethod
    def _split_train_test(in_data, ratio, device):
        np.random.shuffle(in_data)
        num_train_data = int(in_data.shape[0]*ratio)
        train_data = torch.from_numpy(in_data[:num_train_data]).float().to(device)
        test_data = torch.from_numpy(in_data[num_train_data:]).float().to(device)
        return train_data, test_data

    def nn_train(self, verbose=False):
        """
        ____________________________
        Training function to train the NeuralNetwork.
        ____________________________
        """
        self.model.train()

        idx = torch.randperm(self.train_data.size(dim=0))
        torch_data = self.train_data[idx]

        batch_truncation_error = torch.empty((self.batch_size, self.diff_eq.num_y))
        batch_pred = torch.empty((self.batch_size, self.diff_eq.num_y))

        for index, data in enumerate(torch_data):
            # Compute prediction- and truncation- error
            data_temp = np.append(data[:2], data[3:3+self.diff_eq.num_y])
            data_temp = torch.from_numpy(data_temp).float().to(self.device)
            # batch_pred[index % self.batch_size, :] = self.model(data[: 3 + self.diff_eq.num_y])
            batch_pred[index % self.batch_size, :] = self.model(data_temp)
            batch_truncation_error[index % self.batch_size, :] = _local_truncation_error(data, self.diff_eq.func,
                                                                                         self.diff_eq.num_y)

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

                if verbose:
                    loss, current = loss.item(), index+1
                    print(f"loss:{loss:>7f} [{current:>5d}/{self.train_data.shape[0]:>5d}]")

        rest = self.train_data.shape[0] % self.batch_size
        if rest != 0:
            loss = self.loss_fn(batch_pred[:rest], batch_truncation_error[:rest])

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if verbose:
                loss, current = loss.item(), self.train_data.shape[0]
                print(f"loss:{loss:>7f} [{current:>5d}/{self.train_data.shape[0]:>5d}]")

    def nn_test(self, verbose=False):
        """
        ____________________________
        Testing function to test the training of the NeuralNetwork.
        ____________________________
        """
        self.model.eval()
        idx = torch.randperm(self.test_data.size(dim=0))
        torch_data = self.train_data[idx]
        with torch.no_grad():

            batch_truncation_error = torch.empty((self.test_data.size(dim=0), self.diff_eq.num_y))
            batch_pred = torch.empty((self.test_data.size(dim=0), self.diff_eq.num_y))

            for index, data in enumerate(torch_data):
                # Compute prediction- and truncation- error
                # batch_pred[index, :] = self.model(data[: 3 + self.diff_eq.num_y])
                data_temp = np.append(data[:2], data[3:3+self.diff_eq.num_y])
                data_temp = torch.from_numpy(data_temp).float().to(self.device)
                batch_pred[index, :] = self.model(data_temp)
                batch_truncation_error[index, :] = _local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y)

            test_loss = 0
            for i in range(self.diff_eq.num_y):
                test_loss = test_loss + self.loss_fn(batch_pred[:, i], batch_truncation_error[:, i]).float()

            test_loss = test_loss / self.diff_eq.num_y

            if not self.test_loss:
                self.test_loss = [test_loss]
            else:
                self.test_loss.append(test_loss)
            if verbose:
                test_loss, current = test_loss.item(), self.test_data.shape[0]
                print(f"test loss:{test_loss:>7f} [{current:>5d}/{self.test_data.shape[0]:>5d}]")

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def plot_loss(self, semilogy=False):
        epoch = np.arange(1, len(self.test_loss)+1)
        if semilogy:
            plt.semilogy(epoch, self.test_loss)
        else:
            plt.plot(epoch, self.test_loss)
        plt.title("Test Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()


def _local_truncation_error(data, func, num_y):
    # R function
    data = data.cpu()
    y_first = data[3:3+num_y]
    y_second = data[3+num_y:]
    return 1/data[2]**2 * (y_second - y_first - data[2] * func(data[0], y_first))


def main():
    diff_eq = gd.Diff_eq_2(t_0=0, t_end=10, y_0=[2, 4])
    in_data = np.load("outfile_exempel.npy")
    batch_size = 500

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork(diff_eq.num_y).to(device)
    train = TrainAndTest(model=model, diff_eq=diff_eq, in_data=in_data,
                         batch_size=batch_size, device=device, lr=1e-3)
    for i in range(50):
        print("____________________")
        print("epoch:{}".format(i + 1))
        print("____________________")
        train.nn_train()
        train.nn_test()

    train.save_model("eq_1_model_50_no_noise.pth")
    train.plot_loss()


if __name__ == '__main__':
    main()