import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import differentialequations as de


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
            nn.Linear(2+num_y, 80),  # 2 = len([xi, xi+1])
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


class TrainerTester:
    """
    ____________________________
    The TrainerTester class.
    Trains and tests the NeuralNetwork model.
    ____________________________
    """
    def __init__(self, model, diff_eq, in_data, batch_size, device, train_ratio=0.85, lr=1e-3):
        self.model = model
        self.diff_eq = diff_eq
        self.train_data, self.test_data = self._split_train_test(in_data, train_ratio, device)
        self.batch_size = batch_size
        self.loss_fn = nn.L1Loss(reduction="mean")
        self.test_loss = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    @ staticmethod
    def _split_train_test(in_data, train_ratio, device):
        np.random.shuffle(in_data)
        num_train_data = int(len(in_data)*train_ratio)
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
        idx = torch.randperm(self.train_data.size(0))
        train_data = self.train_data[idx]
        train_data = torch.split(train_data, self.batch_size)
        processed_data = 0

        for batch in train_data:

            prediction = torch.empty((len(batch), self.diff_eq.num_y))
            target = torch.empty((len(batch), self.diff_eq.num_y))

            for index, data in enumerate(batch):
                # Compute prediction- and truncation- error
                model_data = data[:2+self.diff_eq.num_y]
                prediction[index, :] = self.model(model_data)
                target[index, :] = _local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y)
                processed_data += 1

            loss = self.loss_fn(prediction, target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                loss, current = loss.item(), processed_data
                print(f"loss:{loss:>7f} [{current:>5d}/{len(self.train_data):>5d}]")

    def nn_test(self, verbose=False):
        """
        ____________________________
        Testing function to test the training of  the NeuralNetwork.
        ____________________________
        """
        self.model.eval()
        with torch.no_grad():

            prediction = torch.empty((len(self.test_data), self.diff_eq.num_y))
            target = torch.empty((len(self.test_data), self.diff_eq.num_y))

            for index, data in enumerate(self.test_data):
                # Compute prediction- and truncation- error
                model_data = data[:2+self.diff_eq.num_y]
                prediction[index, :] = self.model(model_data)
                target[index, :] = _local_truncation_error(data, self.diff_eq.func, self.diff_eq.num_y)

            test_loss = self.loss_fn(prediction, target)
            self.test_loss.append(test_loss)

            if verbose:
                test_loss, current = test_loss.item(), len(self.test_data)
                print(f"test loss:{test_loss:>7f} [{current:>5d}/{len(self.test_data):>5d}]")

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def plot_loss(self, title="Test Loss"):
        epoch = np.arange(1, len(self.test_loss)+1)
        plt.plot(epoch, self.test_loss)
        plt.title(title)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()


def _local_truncation_error(data, func, num_y):
    # R function
    data = data.cpu()
    h = data[1] - data[0]
    y_first = data[2:2+num_y]
    y_second = data[2+num_y:]
    return 1/h**2 * (y_second - y_first - h * func(data[0], y_first))


def main():
    # Properties of training & test data
    t_0 = 0
    t_end = 10
    y_0 = [1, 2]
    number_t = 100
    noise = 0

    # Construct data
    diff_eq = de.VanDerPol(t_0=t_0, t_end=t_end, y_0=y_0)
    t_points = de.create_random_t(t_0, t_end, number_t=number_t)
    data = diff_eq.integrate(t_points=t_points, noise_level=noise)
    in_data = diff_eq.reshape_data(data)

    # Properties of model
    epochs = 10
    batch_size = 500
    lr = 1e-4
    save_file = "test.pth"

    device = "cpu"
    print(f"Using {device} device")

    # Building, training and testing model
    model = NeuralNetwork(diff_eq.num_y).to(device)
    train = TrainerTester(model=model, diff_eq=diff_eq, in_data=in_data, batch_size=batch_size, device=device, lr=lr)
    for i in range(epochs):
        print("____________________")
        print("epoch:{}".format(i + 1))
        print("____________________")
        train.nn_train()
        train.nn_test()

    train.save_model(save_file)
    train.plot_loss()


if __name__ == '__main__':
    main()