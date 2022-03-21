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


def fix_euler_method(model, t0, t_end, y0, h, diff_eq, device, deep=True):
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
        temp0 = np.array([y[-1][j] for j in range(diff_eq.num_y)])
        temp1 = np.array([t[-1], t[-1] + h])
        temp1 = np.append(temp1, temp0)
        data = torch.tensor(temp1).to(device).float()
        nn_e = model(data).cpu()
        nn_e = nn_e.detach().numpy()
        norm = np.amax(abs(nn_e)) * h**2

        if norm < tol:
            h = (tol / norm)**(1/2) * h * 0.9

        while norm > tol:
            i += 1
            h = (tol / norm)**(1/2) * h * 0.9
            temp0 = np.array([y[-1][j] for j in range(diff_eq.num_y)])
            temp1 = np.array([t[-1], t[-1]+h])
            temp1 = np.append(temp1, temp0)
            data = torch.tensor(temp1).to(device).float()
            nn_e = model(data).cpu()
            nn_e = nn_e.detach().numpy()
            norm = np.amax(abs(nn_e)) * h**2

        t.append(t[-1]+h)
        y_temp = np.array(y[-1])
        y_temp = y_temp + h*diff_eq.func(t[-1], y_temp) + h**2 * nn_e
        y.append(list(y_temp))

    t = np.array(t)
    y = np.array(y)
    return t, y, i


def main():
    y0 = [1, 2]
    t0 = 0
    t_end = 25
    diff_eq = Diff_eq_1(t0, t_end, y0)

    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load("eq_1_model_50_adam_batch.pth"))
    model.eval()

    t_dem_adap, y_dem_adap, i = adaptive_euler(model=model, t0=t0, t_end=t_end, y0=y0, h=0.1, tol=0.0001, diff_eq=diff_eq,
                                            device=device)

    t_dem_fix, y_dem_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=(t_end-t0)/(len(t_dem_adap)+i), diff_eq=diff_eq,
                                            device=device)
    t_euler_fix, y_euler_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=(t_end-t0)/(len(t_dem_adap)+i), diff_eq=diff_eq,
                                                device=device, deep=False)
    print("t DEM adap", len(t_dem_adap))
    print("t DEM fix", len(t_dem_fix))
    print("t Euler fix", len(t_euler_fix))

    t_points = np.arange(t0, t_end, (t_end-t0) / (len(t_dem_adap)+i))
    data_integrate = diff_eq.integrate(t_points=t_points)

    y_rel_dem_adap = diff_eq.integrate(t_points=t_dem_adap)
    rel_e_dem_adap1 = np.abs(y_dem_adap[:, 0] - y_rel_dem_adap[:, 1])
    rel_e_dem_adap2 = np.abs(y_dem_adap[:, 1] - y_rel_dem_adap[:, 2])

    rel_e_dem_fix1 = np.abs(y_dem_fix[:, 0] - data_integrate[:, 1])
    rel_e_dem_fix2 = np.abs(y_dem_fix[:, 1] - data_integrate[:, 2])

    rel_e_euler_fix1 = np.abs(y_euler_fix[:, 0] - data_integrate[:, 1])
    rel_e_euler_fix2 = np.abs(y_euler_fix[:, 1] - data_integrate[:, 2])

    # ref_euler = np.abs(y_fix_euler[:-1] - data_integrate[:, 1])
    plt.plot(t_dem_adap, rel_e_dem_adap1, label="DEM adaptive y1")
    plt.plot(t_dem_adap, rel_e_dem_adap2, label="DEM adaptive y2")

    plt.plot(t_dem_fix, rel_e_dem_fix1, "--", label="DEM fix y1")
    plt.plot(t_dem_fix, rel_e_dem_fix2, "--", label="DEM fix y2")

    # plt.plot(t_euler_fix, rel_e_euler_fix1, ".", label="Euler fix y1")
    # plt.plot(t_euler_fix, rel_e_euler_fix2, ".", label="Euler fix y2")
    plt.title("Relative error")
    plt.legend()
    plt.show()

    plt.plot(t_dem_adap, y_dem_adap[:, 0], "o--", markerfacecolor='none', label='DEM adaptive y1')
    plt.plot(t_dem_adap, y_dem_adap[:, 1], "o--", markerfacecolor='none', label='DEM adaptive y2')

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