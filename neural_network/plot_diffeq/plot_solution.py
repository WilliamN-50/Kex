import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from neural_network.model import diffeqnetworkmodel1 as model1, diffeqnetworkmodel2 as model2
from neural_network.model import differentialequations as deq


def euler_method(model, t_0, t_end, y_0, h, diff_eq, deep=True):
    """
    ____________________________
    Solves the IVP using Euler method or deep Euler method.
    ____________________________
    """
    t = np.arange(t_0, t_end, h)
    y = np.empty((len(t), diff_eq.num_y))
    y[0, :] = y_0

    for i in range(len(t)-1):
        func = diff_eq.func(t[i], y[i, :])
        if deep:
            nn_e = calc_nn_residual(model, t[i], h, y[i, :], func)
            y[i+1, :] = y[i, :] + h * func + h**2 * nn_e
        else:
            y[i+1, :] = y[i, :] + h * func
    return t, y


def adaptive_euler(model, t_0, t_end, y_0, h, tol, diff_eq):
    """
    ____________________________
    Solves the IVP using adaptive Euler method.
    ____________________________
    """
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
    """
    ____________________________
    Solves the IVP using implicit Euler or deep implicit Euler method.
    ____________________________
    """
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
        nn_e = calc_nn_residual(model, t_1, h, y_1, func)

        # Implicit Euler + Secant Method
        y_2_guess = y[i, :] + h * func + h**2 * nn_e  # Step using DEM
        y_temp = _secant_method(y_1, y_2_guess, _implicit_euler_func, tol=tol, max_iter=max_iter)
        y[i+1, :] = y_temp
    return t, y


def adaptive_implicit_euler(model, t_0, t_end, y_0, h, tol, diff_eq, secant_tol=0.01, max_iter=1000):
    """
    ____________________________
    Solves the IVP using adaptive implicit Euler.
    ____________________________
    """
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
    """
    ____________________________
    Computes the norm for the adaptive step-size method.
    ____________________________
    """
    nn_e = calc_nn_residual(model, t, h, y, func)
    norm = np.amax(np.abs(nn_e)) * h**2
    return norm, nn_e


def calc_nn_residual(model, t, h, y, func):
    """
    ____________________________
    Computes the output of NeuralNetworkModel.
    ____________________________
    """
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
    residual = y_second - y_first - h * diff_eq.func(t_first, y_first)
    norm = np.amax(np.abs(residual))
    return norm


def plot_diagram(t, y, num_y, label, marker):
    for i in range(num_y):
        plt.plot(t, y[:, i], marker, label=label + str(i+1), markerfacecolor='none')


def plot_diagram_hamiltonian(t, y, num_y, label, marker, title_p="p", title_q="q", log=False):
    """
    ____________________________
    Plots the absolute error for a hamiltonian system.
    ____________________________
    """
    plt.subplot(1, 2, 1)
    for i in range(num_y//2):
        plt.plot(t, y[:, i], marker, label=label + title_q + str(i+1))
    plt.title("Absolute Error of " + title_q)
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.legend()
    if log:
        plt.yscale("log")

    plt.subplot(1, 2, 2)
    for i in range(num_y//2, num_y):
        plt.plot(t, y[:, i], marker, label=label + title_p + str(i+1-num_y//2))
    plt.title("Absolute Error of " + title_p)
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.legend()
    if log:
        plt.yscale("log")


def plot_diagram_phase_2d(y, label, marker, title_p=" p", title_q=" q"):
    """
    ____________________________
    Plots a phase diagram.
    ____________________________
    """
    plt.subplot(1, 2, 1)
    plt.plot(y[:, 0], y[:, 1], marker, label=label + title_q)
    plt.title(title_q)
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y[:, 2], y[:, 3], marker, label=label + title_p)
    plt.title(title_p)
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.legend()


def comp_abs_error(y_prediction, y_target):
    """
    ____________________________
    Computes the absolute error of the prediction and target.
    ____________________________
    """
    rel_error = np.abs(y_prediction - y_target)
    return rel_error


def comp_rel_error(y_prediction, y_target):
    """
    ____________________________
    Computes the relative error of the prediction and target.
    ____________________________
    """
    rel_error = np.divide(np.abs(y_prediction - y_target), np.abs(y_target))
    return rel_error


#Example
def main():
    # Properties of differential equation
    t_0 = 0
    t_end = 30
    y_0 = [0.5, 0, 0, np.sqrt(3)]
    diff_eq = deq.Kepler(t_0, t_end, y_0)

    # Load model
    filename = "model2_residual.pth"
    model = model2.NeuralNetworkModel2(diff_eq.num_y)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Construct data (Adaptive)
    h_0 = 0.01
    tol = 0.001
    t_dem_adap, y_dem_adap, h, n_comp = adaptive_euler(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_0, tol=tol,
                                                       diff_eq=diff_eq)
    adap_data = diff_eq.integrate(t_points=t_dem_adap)
    t_idem_adap, y_idem_adap, ih, in_comp = adaptive_implicit_euler(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_0, tol=tol,
                                                       diff_eq=diff_eq)
    iadap_data = diff_eq.integrate(t_points=t_idem_adap)


    # Construct data (Fix)
    h_fix = min(h)  # When comparing with adaptive step-size method
    # h_fix = h_0  # When not comparing with adaptive step-size method
    t_dem_fix, y_dem_fix = euler_method(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_fix, diff_eq=diff_eq,
                                        deep=True)
    t_idem_fix, y_idem_fix = implicit_euler(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_fix, diff_eq=diff_eq,
                                        deep=True)
    ifix_data = diff_eq.integrate(t_points=t_idem_fix)
    t_euler_fix, y_euler_fix = euler_method(model=model, t_0=t_0, t_end=t_end, y_0=y_0, h=h_fix, diff_eq=diff_eq,
                                            deep=False)
    fix_data = diff_eq.integrate(t_points=t_dem_fix)

    # Print the number of steps taken
    print("t DEM adap", len(t_dem_adap))
    print("t DEM fix", len(t_dem_fix))
    print("t Euler fix", len(t_euler_fix))
    print("t Implicit fix", len(t_idem_fix))

    # Construct relative data
    rel_dem_adap = comp_abs_error(y_dem_adap, adap_data[:, 1:])
    rel_dem_fix = comp_abs_error(y_dem_fix, fix_data[:, 1:])
    # rel_euler_fix = comp_abs_error(y_euler_fix, fix_data[:, 1:])
    rel_idem_fix = comp_abs_error(y_idem_fix, ifix_data[:, 1:])
    rel_idem_adap = comp_abs_error(y_idem_adap, iadap_data[:, 1:])

    # plot_diagram(t_dem_adap, rel_dem_adap, diff_eq.num_y, label="DEM adaptive y", marker="-")
    # plot_diagram(t_dem_fix, rel_dem_fix, diff_eq.num_y, label="DEM fix y", marker="--")
    # plot_diagram(t_euler_fix, rel_euler_fix, diff_eq.num_y, label="Euler fix y", marker=".")
    # plt.yscale("log")
    # plt.xlabel("t")
    # plt.ylabel("Error in log")
    # plt.title("Absolute Error of the ODE")
    # plt.legend()
    # plt.show()

    # plot_diagram(t_dem_adap, y_dem_adap, diff_eq.num_y, label="DEM adaptive y", marker="--.")
    # plot_diagram(t_dem_fix, y_dem_fix, diff_eq.num_y, label="DEM fix y", marker="--")
    # plot_diagram(fix_data[:, 0], fix_data[:, 1:], diff_eq.num_y, label="Reference y", marker="")
    # plt.xlabel("t")
    # plt.ylabel("y")
    # plt.title("Solution of the ODE")
    # plt.legend()
    # plt.show()

    # Use for Kepler
    plot_diagram_hamiltonian(t_dem_fix, rel_dem_fix, diff_eq.num_y, "DEM fix", marker="-", log=True)
    plot_diagram_hamiltonian(t_idem_fix, rel_idem_fix, diff_eq.num_y, "DIEM fix", marker="--", log=True)
    # plot_diagram_hamiltonian(t_dem_adap, rel_dem_adap, diff_eq.num_y, "DEM adaptive", marker="-", log=True)
    # plot_diagram_hamiltonian(t_idem_adap, rel_idem_adap, diff_eq.num_y, "DIEM adap", marker="--", log=True)
    plt.show()

    plot_diagram_phase_2d(y_dem_fix, "DEM fix", marker="--")
    plot_diagram_phase_2d(y_idem_fix, "DIEM fix", marker="--.")
    # plot_diagram_phase_2d(y_dem_adap, "DEM adaptive", marker="--")
    # plot_diagram_phase_2d(y_idem_adap, "DIEM adaptive", marker="--.")
    plot_diagram_phase_2d(fix_data[:, 1:], "Reference", marker="")
    plt.show()


if __name__ == '__main__':
    main()
