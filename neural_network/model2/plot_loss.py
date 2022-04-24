import numpy as np
import matplotlib.pyplot as plt


def plot_loss(loss):
    epochs = np.arange(1, len(loss)+1)
    plt.plot(epochs, loss, label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


def main():
    filename = "loss_model2_LinearODE1_100p_500batch_75ep_lr1e-4 (1).npy"
    loss_data = np.load(filename)
    plot_loss(loss_data)
    plt.legend()
    plt.title("Test loss of Linear ODE, model 2")
    plt.show()

    plt.plot()


if __name__ == '__main__':
    main()
