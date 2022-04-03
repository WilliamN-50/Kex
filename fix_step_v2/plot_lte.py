import torch
import numpy as np
import matplotlib.pyplot as plt
import differentialequations as deq
import diffeqnetwork as den


def model_lte(model, in_data, func):
    """
    ____________________________
    Calculates the local truncation error of the NeuralNetwork model and Euler forward.
    ____________________________
    """
    nn_lte = np.empty((len(in_data)-1, len(in_data[0])-1))
    lte = np.empty(nn_lte.shape)

    for i in range(len(in_data)-1):
        t_first = in_data[i, 0]
        t_second = in_data[i+1, 0]
        y_first = in_data[i, 1:]
        y_second = in_data[i+1, 1:]
        data = np.concatenate((t_first, t_second, y_first, y_second), axis=None)

        torch_data = torch.tensor(data[:2+len(y_first)]).float()
        nn_e = model(torch_data)
        nn_lte[i, :] = nn_e.detach().numpy()
        lte[i, :] = den.euler_local_truncation_error(data, func, len(y_first))

    return nn_lte, lte


def exact_lte(in_data, func):
    """
    ____________________________
    Calculates the local truncation error of Euler forward, implicit Euler, and Euler-Cromer.
    ____________________________
    """
    lte_forward_euler = np.empty((len(in_data)-1, len(in_data[0])-1))
    lte_implicit_euler = np.empty(lte_forward_euler.shape)
    lte_euler_cromer = np.empty(lte_forward_euler.shape)

    num_y = len(in_data[0, 1:])
    idx_x = np.array([i for i in range(num_y//2)])
    idx_v = np.array([i for i in range(num_y//2, num_y)])

    for i in range(len(in_data)-1):
        t_first = in_data[i, 0]
        t_second = in_data[i+1, 0]
        h = t_second - t_first
        y_first = in_data[i, 1:]
        y_second = in_data[i+1, 1:]

        lte_fe = 1 / h**2 * (y_second - y_first - h * func(t_first, y_first))
        lte_ie = 1 / h**2 * (y_second - y_first - h * func(t_second, y_second))
        lte_ec_v = 1 / h**2 * (y_second[idx_v] - y_first[idx_v] - h * func(t_first, y_first)[idx_v])
        lte_ec_x = 1 / h**2 * (y_second[idx_x] - y_first[idx_x] - h * func(t_first, y_second)[idx_x])
        lte_ec = np.concatenate((lte_ec_x, lte_ec_v), axis=None)

        lte_forward_euler[i, :] = lte_fe
        lte_implicit_euler[i, :] = lte_ie
        lte_euler_cromer[i, :] = lte_ec

    return lte_forward_euler, lte_implicit_euler, lte_euler_cromer


def plot_lte(t, lte, num_y, label, marker):
    for i in range(num_y):
        plt.plot(t[:-1], lte[:, i], marker, label=f"{label} of y{i+1}")


def plot_multi_lte(t, lte_list, num_y, labels, markers):
    """
    ____________________________
    Plots the local truncation errors in lte_list.
    ____________________________
    """
    for i in range(len(lte_list)):
        plot_lte(t, lte_list[i], num_y, labels[i], markers[i])


def plot_lte_hamiltonian(t, lte, nn_lte, num_y, title_p, title_q):
    """
    ____________________________
    Plots the local truncation errors for a hamiltonian system.
    ____________________________
    """
    plt.subplot(1, 2, 1)
    for i in range(num_y//2):
        plt.plot(t[:-1], lte[:, i], label="lte of p" + str(i+1))
        plt.plot(t[:-1], nn_lte[:, i], "--", label="nn_lte of p" + str(i+1))
    plt.title(title_p)
    plt.xlabel("t values")
    plt.ylabel("Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(num_y//2, num_y):
        plt.plot(t[:-1], lte[:, i], label="lte of q" + str(i+1-num_y//2))
        plt.plot(t[:-1], nn_lte[:, i], "--", label="nn_lte of q" + str(i+1-num_y//2))
    plt.title(title_q)
    plt.xlabel("t values")
    plt.ylabel("Error")
    plt.legend()


def main():
    # Properties of differential equation
    t_0 = 0
    t_end = 25
    y_0 = [1, 2]
    diff_eq = deq.VanDerPol(t_0, t_end, y_0)
    # y_0 = [0.5, 0, 0, np.sqrt(3)]
    # diff_eq = deq.Kepler(t_0, t_end, y_0)

    # Load model
    filename = "test.pth"
    # filename = "../trained_model/Kepler_no_noise_0_10_1000p_100ep_1e3.pth"
    model = den.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Construct data
    h = 0.01
    t = np.arange(t_0, t_end, h)
    y = diff_eq.integrate(t_points=t)

    # Exact lte
    # lte_fe, lte_ie, lte_ec = exact_lte(y, diff_eq.func)
    # lte_exact = [lte_fe, lte_ie, lte_ec]
    # label_exact = ["Euler Forward lte", "Implicit Euler lte", "Euler-Cromer lte"]
    # markers_exact = ["r", "g", "b"]

    # Model lte
    nn_lte, lte = model_lte(model, y, diff_eq.func)
    label_model = ["lte", "nn_lte"]
    markers_model = ["-", "--"]
    lte_model = [lte, nn_lte]

    title = "Local Truncation Error"

    plot_multi_lte(t, lte_model, diff_eq.num_y, label_model, markers_model)
    # plot_multi_lte(t, lte_exact, diff_eq.num_y, label_exact, markers_exact)
    # plot_lte_hamiltonian(t, lte, nn_lte, diff_eq.num_y, "x^dot", "x")

    plt.title(title)
    plt.xlabel("t values")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
