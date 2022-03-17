import matplotlib.pyplot as plt
import numpy as np
import GenerateData as gd
import NN_model
import torch


class Diff_eq_0(gd.DifferentialEquation):
    def func(self, t, y):
        return -y


class Diff_eq_1(gd.DifferentialEquation):
    def func(self, x, y):
        return np.array([y[0] - y[0]*y[1], -y[1] + y[0]*y[1]])


class Diff_eq_2(gd.DifferentialEquation):
    def func(self, t, y):
        return 3/2 * y/(t+1) + np.sqrt(t+1)


def main():
    diff_eq = Diff_eq_1(0, 10, [1, 2])
    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load("eq_1_model_50.pth"))
    model.eval()

    """
    t = np.arange(0, 10, 0.1)
    data_integrate = diff_eq.integrate(t_points=t)

    # plt.plot(data_integrate[:, 0], data_integrate[:, 1])
    # plt.show()
    data_input = diff_eq.reshape_data(data_integrate)
    
    batch_size = 500
    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    nn_tr_te = NN_model.TrainAndTest(model, diff_eq, data_input, batch_size, device, train_ratio=0.85, lr=1e-3)
    for i in range(2):
        print("____________________")
        print("epoch:{}".format(i + 1))
        print("____________________")
        nn_tr_te.nn_train()
        nn_tr_te.nn_test()
    """

    t2 = np.arange(0, 10, 0.15)
    y = np.zeros((2, len(t2)))
    y[:, 0] = [1, 2]
    for i in range(len(t2)-1):
        h = t2[i+1] - t2[i]
        # print(y[i, 0])
        data = torch.tensor([t2[i], t2[i+1], h, y[0, i], y[1, i]]).to(device).float()
        nn_e = model(data).cpu()
        nn_e = nn_e.detach().numpy()
        y[:, i+1] = y[:, i] + h*diff_eq.func(t2[i], y[:, i]) + h**2 * nn_e

    t = np.arange(0, 10, 0.1)
    data_integrate = diff_eq.integrate(t_points=t)

    plt.plot(t2, y[0, :], label='prediction')
    plt.plot(t2, y[1, :], label='prediction')
    plt.plot(data_integrate[:, 0], data_integrate[:, 1], label='target')
    plt.plot(data_integrate[:, 0], data_integrate[:, 2], label='target')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()