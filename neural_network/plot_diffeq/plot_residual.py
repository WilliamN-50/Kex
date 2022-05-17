import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network.model import diffeqnetworkmodel1 as model1, diffeqnetworkmodel2 as model2
from neural_network.model import differentialequations as deq


def model_residual(model, in_data, func):
    """
    ____________________________
    Calculates the output of the NeuralNetworkModel and the residual of the Euler method.
    ____________________________
    """
    num_y = len(in_data[0])-1
    nn_residual = np.empty((len(in_data)-1, num_y))
    residual = np.empty(nn_residual.shape)

    for i in range(len(in_data)-1):
        if type(model) == model1.NeuralNetworkModel1:
            data = deq.reshape_data_model1(in_data[i:i+2])
            torch_data = torch.tensor(data[0, :2+num_y]).float()
            residual[i, :] = model1.euler_residual(data[0], func, num_y)
        else:
            data = deq.reshape_data_model2(in_data[i:i+2], func)
            torch_data = torch.tensor(data[0, :1+2*num_y]).float()
            residual[i, :] = model2.euler_residual(data[0], num_y)
        nn_e = model(torch_data)
        nn_residual[i, :] = nn_e.detach().numpy()
    return nn_residual, residual


def exact_residual(in_data, func):
    """
    ____________________________
    Calculates the residual of Euler forward and implicit Euler.
    ____________________________
    """
    residual_forward_euler = np.empty((len(in_data)-1, len(in_data[0])-1))
    residual_implicit_euler = np.empty(residual_forward_euler.shape)

    for i in range(len(in_data)-1):
        t_first = in_data[i, 0]
        t_second = in_data[i+1, 0]
        h = t_second - t_first
        y_first = in_data[i, 1:]
        y_second = in_data[i+1, 1:]

        residual_fe = 1 / h**2 * (y_second - y_first - h * func(t_first, y_first))
        residual_ie = 1 / h**2 * (y_second - y_first - h * func(t_second, y_second))

        residual_forward_euler[i, :] = residual_fe
        residual_implicit_euler[i, :] = residual_ie

    return residual_forward_euler, residual_implicit_euler


def plot_residual(t, residual, num_y, label, marker):
    for i in range(num_y):
        plt.plot(t[:-1], residual[:, i], marker, label=f"{label} of y{i+1}")


def plot_multi_residual(t, residual_list, num_y, labels, markers):
    """
    ____________________________
    Plots the residual in residual_list.
    ____________________________
    """
    for i in range(len(residual_list)):
        plot_residual(t, residual_list[i], num_y, labels[i], markers[i])


def plot_residual_hamiltonian(t, residual, nn_residual, num_y, title_p, title_q):
    """
    ____________________________
    Plots the residual for a hamiltonian system.
    ____________________________
    """
    plt.subplot(1, 2, 1)
    for i in range(num_y//2):
        plt.plot(t[:-1], residual[:, i], label="reference residual, q" + str(i+1))
        plt.plot(t[:-1], nn_residual[:, i], "--", label="model residual, q" + str(i+1))
    plt.title(title_q)
    plt.xlabel("t")
    plt.ylabel("residual")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(num_y//2, num_y):
        plt.plot(t[:-1], residual[:, i], label="reference residual, p" + str(i+1-num_y//2))
        plt.plot(t[:-1], nn_residual[:, i], "--", label="model residual, p" + str(i+1-num_y//2))
    plt.title(title_p)
    plt.xlabel("t")
    plt.ylabel("residual")
    plt.legend()


# Example
def main():
    # Properties of differential equation
    t_0 = 0
    t_end = 30
    y_0 = [1]
    diff_eq = deq.LinearODE1(t_0, t_end, y_0)

    # Load model
    filename = "model2_residual.pth"
    model = model2.NeuralNetworkModel2(diff_eq.num_y)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Construct data
    h = 0.1
    t = np.arange(t_0, t_end, h)
    y = diff_eq.integrate(t_points=t)

    # Exact residual
    # residual_fe, residual_ie = exact_residual(y, diff_eq.func)
    # residual_exact = [residual_fe, residual_ie]
    # label_exact = ["Euler Forward residual", "Implicit Euler residual"]
    # markers_exact = ["r", "b"]

    # Model residual
    nn_residual, residual = model_residual(model, y, diff_eq.func)
    label_model = ["reference residual", "model residual"]
    markers_model = ["-", "--"]
    residual_model = [residual, nn_residual]
    
    
    # plot_multi_residual(t, residual_exact, diff_eq.num_y, label_exact, markers_exact)
    plot_multi_residual(t, residual_model, diff_eq.num_y, label_model, markers_model)
    plt.xlabel("t")
    plt.ylabel("residual")
    plt.legend()
    plt.show()
    
    # For Kepler
    # plot_residual_hamiltonian(t, residual, nn_residual, diff_eq.num_y, "residual, p, h=0.1, model 1", "residual, q, h=0.1, model 1")
    # plt.show()


if __name__ == '__main__':
    main()
