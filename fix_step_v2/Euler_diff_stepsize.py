import matplotlib.pyplot as plt
import numpy as np
import GenerateData as gd
import NN_model
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

        if deep:
            y[i+1, :] = y[i, :] + h*diff_eq.func(t[i], y[i, :]) + h**2 * nn_e
        else:
            y[i + 1, :] = y[i, :] + h * diff_eq.func(t[i], y[i, :])

    return t, y


def adaptive_euler(model, t0, t_end, y0, h, tol, diff_eq, device):
    t = [t0]
    y = [y0]
    h_list = []

    i = 0
    while t[-1] < t_end - h:
        norm, nn_e = comp_norm(model, t[-1], y[-1], h, diff_eq, device)
        if norm < tol:
            i += 1
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t[-1], y[-1], h, diff_eq, device)

        while norm > tol:
            i += 1
            h = (tol / norm)**(1/2) * h * 0.9
            norm, nn_e = comp_norm(model, t[-1], y[-1], h, diff_eq, device)

        y_temp = np.array(y[-1])
        y_temp = y_temp + h*diff_eq.func(t[-1], y_temp) + h**2 * nn_e
        y.append(list(y_temp))
        t.append(t[-1] + h)
        h_list.append(h)

    t = np.array(t)
    y = np.array(y)
    return t, y, i, h_list


def comp_norm(model, t0, y0, h, diff_eq, device):
    temp0 = np.array([y0[j] for j in range(diff_eq.num_y)])
    temp1 = np.array([t0, t0 + h])
    temp1 = np.append(temp1, temp0)
    data = torch.tensor(temp1).to(device).float()
    nn_e = model(data).cpu()
    nn_e = nn_e.detach().numpy()
    norm = np.amax(abs(nn_e)) * h**2
    # norm = np.sum(abs(nn_e))/diff_eq.num_y * h**2 /2
    return norm, nn_e


def plot_diagram(num_y, x, y, label_title, mark):
    for i in range(num_y):
        plt.plot(x, y[:, i], mark, label=label_title + str(i + 1), markerfacecolor='none')


def comp_rel_error(num_y, y_prediction, y_target):
    rel_error = np.empty((y_prediction.shape[0], y_prediction.shape[1]))
    for i in range(num_y):
        rel_error[:, i] = np.abs(y_prediction[:, i] - y_target[:, i + 1])

    return rel_error


def main():
    y0 = [0.5, 0, 0, np.sqrt(3)]
    # y0=[1, 2]
    t0 = 0
    t_end = 25
    tol = 0.0001
    h_start = 0.1
    diff_eq = gd.Kepler(t_0=t0, t_end=t_end, y_0=y0)
    # diff_eq = gd.LodkaVolterra(t0, t_end, y0)

    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load("../trained_model/Kepler_no_noise_0_10_1000p_75ep.pth"))
    # model.load_state_dict(torch.load("../trained_model/eq_1_model_50_adam_batch.pth"))
    # model.load_state_dict(torch.load("eq_van_der_model_Adam_no_noise_1_2_1000p_100ep_lr5_10_4.pth"))
    model.eval()

    t_dem_adap, y_dem_adap, i, h_list = adaptive_euler(model=model, t0=t0, t_end=t_end, y0=y0, h=h_start, tol=tol,
                                                       diff_eq=diff_eq, device=device)
    h = np.min(h_list)
    # h = 25/(len(t_dem_adap))
    t_dem_fix, y_dem_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=h, diff_eq=diff_eq,
                                            device=device)
    # print(y_dem_fix)
    t_euler_fix, y_euler_fix = fix_euler_method(model=model, t0=t0, t_end=t_end, y0=y0, h=h, diff_eq=diff_eq,
                                                device=device, deep=False)
    print("t DEM adap", len(t_dem_adap))
    print("t DEM fix", len(t_dem_fix))
    print("t Euler fix", len(t_euler_fix))
    print(h)
    print(i)

    data_integrate = diff_eq.integrate(t_points=t_dem_fix)
    y_rel_dem_adap = diff_eq.integrate(t_points=t_dem_adap)

    rel_e_dem_adap = comp_rel_error(diff_eq.num_y, y_dem_adap, y_rel_dem_adap)
    rel_e_dem_fix = comp_rel_error(diff_eq.num_y, y_dem_fix, data_integrate)
    rel_e_euler_fix = comp_rel_error(diff_eq.num_y, y_euler_fix, data_integrate)

    plot_diagram(diff_eq.num_y, t_dem_adap, rel_e_dem_adap, "DEM adaptive y", "")
    plot_diagram(diff_eq.num_y, t_dem_fix, rel_e_dem_fix, "DEM fix y", "--")
    plot_diagram(diff_eq.num_y, t_euler_fix, rel_e_euler_fix, "Euler fix y", ".")
    plt.title("Relative error")
    plt.legend()
    plt.show()

    plot_diagram(diff_eq.num_y, t_dem_adap, y_dem_adap, label_title="DEM adaptive y", mark="o--")
    plot_diagram(diff_eq.num_y, t_dem_fix, y_dem_fix, label_title="DEM fix y", mark="--")
    plot_diagram(diff_eq.num_y, data_integrate[:, 0], data_integrate[:, 1:], label_title="Exact y", mark="")
    plt.title("Solve of the ODE")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()
