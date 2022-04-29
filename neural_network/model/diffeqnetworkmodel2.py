import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from neural_network.model import differentialequations as deq


# Define model
class NeuralNetworkModel2(nn.Module):
    """
    ____________________________
    The NeuralNetwork class.
    Constructs a neural network for predicting the local truncation error of the Euler method.
    Input = (h_i, y_i, f(t_i, y_i))
    ____________________________
    """
    def __init__(self, num_y):
        super(NeuralNetworkModel2, self).__init__()  # Take the init from nn.Module
        self.linear_result_stack = nn.Sequential(
            nn.Linear(1+2*num_y, 80),
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


class TrainerTesterModel2:
    """
    ____________________________
    The TrainerTester class.
    Trains and tests NeuralNetworkModel2.
    ____________________________
    """
    def __init__(self, model, diff_eq, in_data, batch_size, device, train_ratio=0.85, lr=1e-3, random_split=True):
        self.model = model
        self.diff_eq = diff_eq
        self.train_data, self.test_data = self._split_train_test(in_data, train_ratio, device, random_split)
        self.batch_size = batch_size
        self.loss_fn = nn.L1Loss(reduction="mean")
        self.test_loss = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    @ staticmethod
    def _split_train_test(in_data, train_ratio, device, random_split):
        if random_split:
            np.random.shuffle(in_data)
        num_train_data = int(len(in_data)*train_ratio)
        train_data = torch.from_numpy(in_data[:num_train_data]).float().to(device)
        test_data = torch.from_numpy(in_data[num_train_data:]).float().to(device)
        return train_data, test_data

    def nn_train(self, verbose=False):
        """
        ____________________________
        Function to train NeuralNetworkModel2.
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
                model_data = data[:1+2*self.diff_eq.num_y]
                prediction[index, :] = self.model(model_data)
                target[index, :] = euler_local_truncation_error(data.cpu(), self.diff_eq.num_y)
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
        Function to test the training of NeuralNetworkModel2.
        ____________________________
        """
        self.model.eval()
        with torch.no_grad():

            prediction = torch.empty((len(self.test_data), self.diff_eq.num_y))
            target = torch.empty((len(self.test_data), self.diff_eq.num_y))

            for index, data in enumerate(self.test_data):
                # Compute prediction- and truncation- error
                model_data = data[:1+2*self.diff_eq.num_y]
                prediction[index, :] = self.model(model_data)
                target[index, :] = euler_local_truncation_error(data.cpu(), self.diff_eq.num_y)

            test_loss = self.loss_fn(prediction, target)
            self.test_loss.append(test_loss)

            if verbose:
                test_loss, current = test_loss.item(), len(self.test_data)
                print(f"test loss:{test_loss:>7f} [{current:>5d}/{len(self.test_data):>5d}]")

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def plot_loss(self, title="Test Loss"):
        """
        ____________________________
        Plot the loss from the training.
        ____________________________
        """
        epoch = np.arange(1, len(self.test_loss)+1)
        plt.plot(epoch, self.test_loss)
        plt.title(title)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def save_loss(self, filename):
        np.save(filename, self.test_loss)


def euler_local_truncation_error(data, num_y):
    """
    ____________________________
    Calculates the local truncation error of the Euler method.
    ____________________________
    """
    # lte function
    h = data[0]
    y_first = data[1:1+num_y]
    func = data[1+num_y:1+2*num_y]
    y_second = data[1+2*num_y:1+3*num_y]
    return 1/h**2 * (y_second - y_first - h * func)


# Example
def main():
    # Properties of training & test data
    t_0 = 0
    t_end = 10
    t_start = 0
    y_0 = [0.5, 0, 0, np.sqrt(3)]
    number_t = 1000
    noise = 0

    # Construct data
    diff_eq = deq.Kepler(t_0=t_0, t_end=t_end, y_0=y_0)
    t_points = deq.create_random_t(t_start, t_end, number_t=number_t)
    data = diff_eq.integrate(t_points=t_points, noise_level=noise)
    in_data = deq.reshape_data_model2(data, diff_eq.func)

    # Properties of model
    epochs = 1000
    batch_size = 100
    lr = 1e-4
    model_file = "model2_lte.pth"
    loss_file = "model2_loss.npy"

    device = "cpu"
    print(f"Using {device} device")

    # Building, training and testing model
    model = NeuralNetworkModel2(diff_eq.num_y).to(device)
    train = TrainerTesterModel2(model=model, diff_eq=diff_eq, in_data=in_data, batch_size=batch_size, device=device, lr=lr)
    for i in range(epochs):
        print("____________________")
        print("epoch:{}".format(i + 1))
        print("____________________")
        train.nn_train()
        train.nn_test()

    train.save_model(model_file)
    train.save_loss(loss_file)
    train.plot_loss()


if __name__ == '__main__':
    main()
