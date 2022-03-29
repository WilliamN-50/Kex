import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import differentialequations as deq
import diffeqnetwork as den


def fix_euler_method(model, t0, t_end, y0, h, diff_eq, deep=True):
    t = np.arange(t0, t_end, h)
    y = np.empty((len(t), diff_eq.num_y))
    y[0, :] = y0

    for i in range(len(t)-1):
        if deep:
            nn_e = calc_nn_lte(model, t[i], y[i, :], h)
            y[i+1, :] = y[i, :] + h*diff_eq.func(t[i], y[i, :]) + h**2 * nn_e
        else:
            y[i+1, :] = y[i, :] + h * diff_eq.func(t[i], y[i, :])

    return t, y


def adaptive_euler(model, t0, t_end, y0, h, tol, diff_eq):
    t = [t0]
    y = [y0]
    h_list = []

    num_comp_norm = 0  # Number of extra times comp_norm is called for each loop
    while t[-1] < t_end - h:
        norm, nn_e = comp_norm(model, t[-1], y[-1], h)
        if norm < tol:
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t[-1], y[-1], h)
            num_comp_norm += 1

        while norm > tol:
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t[-1], y[-1], h)
            num_comp_norm += 1

        y_next = np.array(y[-1])
        y_next = y_next + h * diff_eq.func(t[-1], y_next) + h**2 * nn_e
        y.append(list(y_next))
        t.append(t[-1] + h)
        h_list.append(h)

    t = np.array(t)
    y = np.array(y)
    h_list = np.array(h_list)
    return t, y, h_list, num_comp_norm


def comp_norm(model, t, y, h):
    nn_e = calc_nn_lte(model, t, y, h)
    norm = np.amax(np.abs(nn_e)) * h**2
    return norm, nn_e


def calc_nn_lte(model, t, y, h):
    data = np.concatenate(([t, t+h], y), axis=0)
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


def comp_rel_error(y_prediction, y_target):
    rel_error = np.abs(y_prediction - y_target)
    return rel_error


def main():
    # Properties of differential equation
    t0 = 0
    t_end = 25
    y0 = [1, 2]
    diff_eq = deq.VanDerPol(t0, t_end, y0)

    # Load model
    filename = "test2.pth"
    # filename = "../trained_model/Kepler_no_noise_0_10_1000p_100ep_1e3.pth"
    model = den.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Construct data (Adaptive)
    h0 = 0.01
    tol = 0.01
    t_dem_adap, y_dem_adap, h, n_comp = adaptive_euler(model=model, t0=t0, t_end=t_end, y0=y0, h=h0, tol=tol,
                                                       diff_eq=diff_eq)
    adap_data = diff_eq.integrate(t_points=t_dem_adap)

    # Construct data (Fix)
    h_fix = 0.01
    t_dem_fix, y_dem_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=h_fix, diff_eq=diff_eq, deep=True)
    t_euler_fix, y_euler_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=h_fix, diff_eq=diff_eq,
                                                deep=False)
    fix_data = diff_eq.integrate(t_points=t_dem_fix)

    # Print the number of steps taken
    print("t DEM adap", len(t_dem_adap))
    print("t DEM fix", len(t_dem_fix))
    print("t Euler fix", len(t_euler_fix))

    # Construct relative data
    rel_dem_adap = comp_rel_error(y_dem_adap, adap_data[:, 1:])
    rel_dem_fix = comp_rel_error(y_dem_fix, fix_data[:, 1:])
    rel_euler_fix = comp_rel_error(y_euler_fix, fix_data[:, 1:])

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


if __name__ == '__main__':
    main()