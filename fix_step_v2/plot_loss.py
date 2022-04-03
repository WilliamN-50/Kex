import numpy as np
import matplotlib.pyplot as plt


def plot_loss(loss):
    epochs = np.arange(1, len(loss)+1)
    plt.plot(epochs, loss, label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


def main():
    filename = "loss2.npy"
    loss_data = np.load(filename)
    plot_loss(loss_data)
    plt.legend()
    plt.show()

    plt.plot()


if __name__ == '__main__':
    main()
