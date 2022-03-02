import torch
from torch import nn
from torchvision.transforms import ToTensor

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()  # Take the init from nn.Module
        self.linear_result_stack = nn.Sequential(
            nn.Linear(3, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
            nn.ReLU()
        )

    def forward(self, x):  # Ett steg beräkning, x = [xi, xi+1, yi]
        result = self.linear_result_stack(x)
        return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    print(model)


if __name__ == '__main__':
    main()
