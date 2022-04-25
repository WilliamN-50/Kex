import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from neural_network.model import diffeqnetworkmodel1 as model1, diffeqnetworkmodel2 as model2
from neural_network.model import differentialequations as deq


def euler_method(model, t_0, t_end, y_0, h, diff_eq, deep=True):
    t = np.arange(t_0, t_end, h)
    y = np.empty((len(t), diff_eq.num_y))
    y[0, :] = y_0

    for i in range(len(t)-1):
        func = diff_eq.func(t[i], y[i, :])
        if deep:
            nn_e = calc_nn_lte(model, t[i], h, y[i, :], func)
            y[i+1, :] = y[i, :] + h * func + h**2 * nn_e
        else:
            y[i+1, :] = y[i, :] + h * func
    return t, y


def adaptive_euler(model, t_0, t_end, y_0, h, tol, diff_eq):
    t = [t_0]
    y = [y_0]
    h_list = []

    num_comp_norm = 0  # Number of extra times comp_norm is called for each loop
    while t[-1] < t_end:
        t_1 = t[-1]
        y_1 = np.array(y[-1])
        func = diff_eq.func(t_1, y_1)
        norm, nn_e = comp_norm(model, t_1, h, y_1, func)
        num_comp_norm += 1
        if norm < tol:
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t_1, h, y_1, func)
            num_comp_norm += 1

        while norm > tol:
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t_1, h, y_1, func)
            num_comp_norm += 1

        y_1 = y_1 + h * func + h**2 * nn_e
        y.append(list(y_1))
        t.append(t_1 + h)
        h_list.append(h)

    del t[-1]
    del y[-1]
    del h_list[-1]
    t = np.array(t)
    y = np.array(y)
    h_list = np.array(h_list)
    return t, y, h_list, num_comp_norm


def _secant_method(y_1, y_2, func, tol=0.01, max_iter=1000):
    itr = 0
    delta = tol + 1

    while delta > tol and itr < max_iter:
        y_temp = y_2
        y_2 = y_2 - func(y_2) * (y_2 - y_1)/(func(y_2)-func(y_1))
        y_1 = y_temp
        delta = np.amax(np.abs(y_2 - y_1))
    return y_2


def implicit_euler(model, t_0, t_end, y_0, h, diff_eq, tol=0.01, max_iter=1000, deep=True):
    t = np.arange(t_0, t_end, h)
    y = np.empty((len(t), diff_eq.num_y))
    y[0, :] = y_0

    if deep:
        def _implicit_euler_func(x):
            return x - y_1 - h * diff_eq.func(t_2, x) + h**2 * nn_e  # Need to have + instead of -
    else:
        def _implicit_euler_func(x):
            return x - y_1 - h * diff_eq.func(t_2, x)

    for i in range(len(t)-1):
        t_1 = t[i]
        t_2 = t[i+1]
        y_1 = y[i, :]
        func = diff_eq.func(t_1, y_1)
        nn_e = calc_nn_lte(model, t_1, h, y_1, func)

        # Implicit Euler + Secant Method
        y_2_guess = y[i, :] + h * func + h**2 * nn_e  # Step using DEM
        y_temp = _secant_method(y_1, y_2_guess, _implicit_euler_func, tol=tol, max_iter=max_iter)
        y[i+1, :] = y_temp
    return t, y


def adaptive_implicit_euler(model, t_0, t_end, y_0, h, tol, diff_eq, secant_tol=0.01, max_iter=1000):
    t = [t_0]
    y = [y_0]
    h_list = []

    def _implicit_euler_func(x):
        return x - y_1 - h * diff_eq.func(t_2, x) + h**2 * nn_e  # Need to have + instead of -

    num_comp_norm = 0  # Number of extra times comp_norm is called for each loop
    while t[-1] < t_end:
        t_1 = t[-1]
        y_1 = np.array(y[-1])
        func = diff_eq.func(t[-1], y_1)
        norm, nn_e = comp_norm(model, t_1, h, y_1, func)
        num_comp_norm += 1
        if norm < tol:
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t_1, h, y_1, func)
            num_comp_norm += 1

        while norm > tol:
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t_1, h, y_1, func)
            num_comp_norm += 1

        # Implicit Euler + Secant Method
        t_2 = t_1 + h
        y_2_guess = y_1 + h * func + h**2 * nn_e  # Step using DEM
        y_2 = _secant_method(y_1, y_2_guess, _implicit_euler_func, tol=secant_tol, max_iter=max_iter)
        y.append(list(y_2))
        t.append(t_2)
        h_list.append(h)

    del t[-1]
    del y[-1]
    del h_list[-1]
    t = np.array(t)
    y = np.array(y)
    h_list = np.array(h_list)
    return t, y, h_list, num_comp_norm


def comp_norm(model, t, h, y, func):
    nn_e = calc_nn_lte(model, t, h, y, func)
    norm = np.amax(np.abs(nn_e)) * h**2
    return norm, nn_e


def calc_nn_lte(model, t, h, y, func):
    if type(model) == model1.NeuralNetworkModel1:
        data = np.concatenate(([t, t+h], y), axis=0)
    else:
        data = np.concatenate(([h], y, func), axis=0)
    data = torch.tensor(data).float()
    nn_e = model(data)
    nn_e = nn_e.detach().numpy()
    return nn_e


def _exact_norm(t_first, y_first, h, diff_eq):
    solution = solve_ivp(diff_eq.func, [t_first, t_first + h], y_first)
    y_second = solution.y[:, -1]
    lte = y_second - y_first - h * diff_eq.func(t_first, y_first)
    norm = np.amax(np.abs(lte))
    return norm


def plot_diagram(t, y, num_y, label, marker):
    for i in range(num_y):
        plt.plot(t, y[:, i], marker, label=label + str(i+1), markerfacecolor='none')


def plot_diagram_hamiltonian(t, y, y_exact, num_y, method_label, title_p="p", title_q="q"):
    """
    ____________________________
    Plots the local truncation errors for a hamiltonian system.
    ____________________________
    """
    plt.subplot(1, 2, 1)
    for i in range(num_y//2):
        plt.plot(t, y_exact[:, i], label=method_label + " exact p" + str(i+1))
        plt.plot(t, y[:, i], "--", label=method_label + " p" + str(i+1))
    plt.title(title_p)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(num_y//2, num_y):
        plt.plot(t, y_exact[:, i], label=method_label + " exact q" + str(i+1-num_y//2))
        plt.plot(t, y[:, i], "--", label=method_label + " q" + str(i+1-num_y//2))
    plt.title(title_q)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()


def plot_diagram_phase_2d(y, y_exact, method_label, title_p="p", title_q="q"):
    plt.subplot(1, 2, 1)
    plt.plot(y_exact[:, 0], y_exact[:, 1], label=method_label + " exact")
    plt.plot(y[:, 0], y[:, 1], "--", label=method_label + " model")
    plt.title(title_p)
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_exact[:, 2], y_exact[:, 3], label=method_label + " exact")
    plt.plot(y[:, 2], y[:, 3], "--", label=method_label + " model")
    plt.title(title_q)
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.legend()


def comp_abs_error(y_prediction, y_target):
    rel_error = np.abs(y_prediction - y_target)
    return rel_error


def comp_rel_error(y_prediction, y_target):
    rel_error = np.divide(np.abs(y_prediction - y_target), np.abs(y_target))
    return rel_error


def main():
    # Properties of differential equation
    t_0 = 0
    t_end = 30
    # y_0 = [1, 2]
    # diff_eq = deq.VanDerPol(t_0, t_end, y_0)
    y_0 = [0.5, 0, 0, np.sqrt(3)]
    diff_eq = deq.Kepler(t_0, t_end, y_0)

    # Load model
    filename = "../../trained_model/model2/model2_Kepler_1000p_500batch_75ep_lr5e-4.pth"
    # filename = "../../trained_model/model 1 input[x, y]/Kepler_no_noise_0_10_1000p_75ep.pth"
    model = model2.NeuralNetworkModel2(diff_eq.num_y)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Construct data (Adaptive)
    h_0 = 0.01
    tol = 0.001
    t_dem_adap, y_dem_adap, h, n_comp = adaptive_euler(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_0, tol=tol,
                                                       diff_eq=diff_eq)
    adap_data = diff_eq.integrate(t_points=t_dem_adap)

    # Construct data (Fix)
    h_fix = min(h)
    t_dem_fix, y_dem_fix = euler_method(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_fix, diff_eq=diff_eq,
                                        deep=True)
    t_euler_fix, y_euler_fix = euler_method(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_fix, diff_eq=diff_eq,
                                            deep=False)
    fix_data = diff_eq.integrate(t_points=t_dem_fix)

    # Print the number of steps taken
    print("t DEM adap", len(t_dem_adap))
    print("t DEM fix", len(t_dem_fix))
    print("t Euler fix", len(t_euler_fix))

    # Construct relative data
    rel_dem_adap = comp_abs_error(y_dem_adap, adap_data[:, 1:])
    rel_dem_fix = comp_abs_error(y_dem_fix, fix_data[:, 1:])
    # rel_euler_fix = comp_abs_error(y_euler_fix, fix_data[:, 1:])

    plot_diagram(t_dem_adap, rel_dem_adap, diff_eq.num_y, label="DEM adaptive y", marker="-")
    plot_diagram(t_dem_fix, rel_dem_fix, diff_eq.num_y, label="DEM fix y", marker="--")
    # plot_diagram(t_euler_fix, rel_euler_fix, diff_eq.num_y, label="Euler fix y", marker=".")
    plt.title("Relative error")
    plt.legend()
    plt.show()

    plot_diagram(t_dem_adap, y_dem_adap, diff_eq.num_y, label="DEM adaptive y", marker="o--")
    plot_diagram(t_dem_fix, y_dem_fix, diff_eq.num_y, label="DEM fix y", marker="--")
    plot_diagram(fix_data[:, 0], fix_data[:, 1:], diff_eq.num_y, label="Exact y", marker="")
    plt.title("Solution of the ODE")
    plt.legend()
    plt.show()

    # plot_diagram_hamiltonian(t_dem_fix, y_dem_fix, fix_data[:, 1:], diff_eq.num_y, "Euler Forward")
    plot_diagram_phase_2d(y_dem_fix, fix_data[:, 1:], "Euler Forward")
    plt.show()
    print(sum(h)/len(h))


if __name__ == '__main__':
    main()
