# Solving Differential Equations using Adaptive Size with Neural Network
The Source Code of the bachelor degree project.
Two kinds of the neural network models are implemented. All of them have 8 hidden layers in addition to the 
indispensable input and output layers. Each hidden layer has $80$ neurons and uses the $ReLU$ function as the activation
function.

## Model
1. [**Model 1**]() uses the input x = (t_i, t_j, y_i).
2. [**Model 2**]() uses the input x = (h_i, y_i, f(t_i, y_i)).

## Training of the model
The loss function used in the training is L1, and the optimization algorithm selected is Adam.

## Example of the differential equation
There are three example in each model:
1. The Linear ODE
2. The Van der Pol equation
3. The Kepler equation
