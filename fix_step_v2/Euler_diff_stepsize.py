import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import differentialequations as de
import diffeqnetwork as deq
import torch


def fix_euler_method(model, t0, t_end, y0, h, diff_eq, device, deep=True):
    # print(h)
    t = np.arange(t0, t_end, h)
    y = np.empty((t.shape[0], diff_eq.num_y))
    y[0, :] = y0

    for i in range(len(t)-1):

        t_first = t[i]
        t_second = t[i+1]
        temp_t = np.array([t_first, t_second])
        temp = np.append(temp_t, y[i, :])

        data = torch.tensor(temp).to(device).float()
        nn_e = model(data).cpu()
        nn_e = nn_e.detach().numpy()
        # norm, nn_e = r_norm(t[i], y[i], h, diff_eq)
        if deep:
            y[i+1, :] = y[i, :] + h*diff_eq.func(t[i], y[i, :]) + h**2 * nn_e
        else:
            y[i + 1, :] = y[i, :] + h * diff_eq.func(t[i], y[i, :])

    return t, y


def adaptive_euler(model, t0, t_end, y0, h, tol, diff_eq, device):
    t = [t0]
    y = [y0]
    i = 0
    while t[-1] < t_end - h:
        norm, nn_e = comp_norm(model, t[-1], y[-1], h, diff_eq, device)
        # norm, nn_e = r_norm(t[-1], y[-1], h, diff_eq)
        h = (tol / norm)**(1/2) * h * 0.9
        norm, nn_e = comp_norm(model, t[-1], y[-1], h, diff_eq, device)
        # norm, nn_e = r_norm(t[-1], y[-1], h, diff_eq)

        while norm > tol:
            i += 1
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t[-1], y[-1], h, diff_eq, device)
            # norm, nn_e = r_norm(t[-1], y[-1], h, diff_eq)

        y_temp = np.array(y[-1])
        y_temp = y_temp + h*diff_eq.func(t[-1], y_temp) + h**2 * nn_e
        y.append(list(y_temp))
        t.append(t[-1] + h)
        # print(h)

    t = np.array(t)
    y = np.array(y)
    return t, y, i


def comp_norm(model, t0, y0, h, diff_eq, device):
    temp0 = np.array([y0[j] for j in range(diff_eq.num_y)])
    temp1 = np.array([t0, t0 + h])
    temp1 = np.append(temp1, temp0)
    data = torch.tensor(temp1).to(device).float()
    nn_e = model(data).cpu()
    nn_e = nn_e.detach().numpy()
    norm = np.amax(np.abs(nn_e)) * h**2
    return norm, nn_e


def r_norm(t_first, y_first, h, diff_eq):
    sol = scipy.integrate.solve_ivp(diff_eq.func, [t_first, t_first + h], y_first)
    y_second = sol.y[:, -1]
    r_temp = 1/h**2 * (y_second - y_first - h * diff_eq.func(t_first, y_first))
    norm = np.linalg.norm(r_temp) * h**2
    return norm, r_temp


def main():
    y0 = [1, 2]
    t0 = 0
    t_end = 50
    diff_eq = de.VanDerPol(t0, t_end, y0)

    device = "cpu"
    model = deq.NeuralNetwork(diff_eq.num_y)
    # model.load_state_dict(torch.load("../trained_model/eq_1_model_trained_gamla.pth"))
    model.load_state_dict(torch.load("test2.pth"))
    model.eval()

    # t_dem_adap, y_dem_adap, i = adaptive_euler(model=model, t0=t0, t_end=t_end, y0=y0, h=0.1, tol=0.01, diff_eq=diff_eq,
    #                                            device=device)

    # h = (t_end-t0)/(len(t_dem_adap)+i)
    h = 0.01

    t_dem_fix, y_dem_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=h, diff_eq=diff_eq,
                                            device=device)
    t_euler_fix, y_euler_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=h, diff_eq=diff_eq,
                                                device=device, deep=True)
    # print("t DEM adap", len(t_dem_adap))

    print("t DEM fix", len(t_dem_fix))
    # print((t_end-t0)/(len(t_dem_adap)+i))
    # print("t Euler fix", len(t_euler_fix))

    data_integrate = diff_eq.integrate(t_points=t_dem_fix)
    # y_rel_dem_adap = diff_eq.integrate(t_points=t_dem_adap)

    # rel_e_dem_adap1 = np.abs(y_dem_adap[:, 0] - y_rel_dem_adap[:, 1])
    # rel_e_dem_adap2 = np.abs(y_dem_adap[:, 1] - y_rel_dem_adap[:, 2])

    rel_e_dem_fix1 = np.abs(y_dem_fix[:, 0] - data_integrate[:, 1])
    rel_e_dem_fix2 = np.abs(y_dem_fix[:, 1] - data_integrate[:, 2])

    rel_e_euler_fix1 = np.abs(y_euler_fix[:, 0] - data_integrate[:, 1])
    rel_e_euler_fix2 = np.abs(y_euler_fix[:, 1] - data_integrate[:, 2])

    # ref_euler = np.abs(y_fix_euler[:-1] - data_integrate[:, 1])
    # plt.plot(t_dem_adap, rel_e_dem_adap1, label="DEM adaptive y1")
    # plt.plot(t_dem_adap, rel_e_dem_adap2, label="DEM adaptive y2")

    plt.plot(t_dem_fix, rel_e_dem_fix1, "--", label="DEM fix y1")
    plt.plot(t_dem_fix, rel_e_dem_fix2, "--", label="DEM fix y2")

    # plt.plot(t_euler_fix, rel_e_euler_fix1, ".", label="Euler fix y1")
    # plt.plot(t_euler_fix, rel_e_euler_fix2, ".", label="Euler fix y2")
    plt.title("Relative error")
    plt.legend()
    plt.show()

    # plt.plot(t_dem_adap, y_dem_adap[:, 0], "o--", markerfacecolor='none', label='DEM adaptive y1')
    # plt.plot(t_dem_adap, y_dem_adap[:, 1], "o--", markerfacecolor='none', label='DEM adaptive y2')

    plt.plot(t_dem_fix, y_dem_fix[:, 0], "--", label="DEM fix y1")
    plt.plot(t_dem_fix, y_dem_fix[:, 1], "--", label="DEM fix y2")

    # plt.plot(t_euler_fix, y_euler_fix[:, 0], ".", label="Euler fix y1")
    # plt.plot(t_euler_fix, y_euler_fix[:, 1], ".", label="Euler fix y2")

    plt.plot(data_integrate[:, 0], data_integrate[:, 1], label='Exact y1')
    plt.plot(data_integrate[:, 0], data_integrate[:, 2], label='Exact y2')
    plt.title("Solve of the ODE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()