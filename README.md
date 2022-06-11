# Solving Differential Equations using Adaptive Size with Neural Network
This code is made for a bachelor degree project. 

## Model
The nerual networks designed in this code predict the residual for a numerical method. 
Two feed-forward neural networks with different input layers are implemented using pytorch. 
Both of them have 8 hidden layers with 80 neurons in each layer. The activation function ReLU is used between the layers.
1. Model 1 uses the input x = (t_i, t_j, y_i).
2. Model 2 uses the input x = (h_i, y_i, f(t_i, y_i)).

Here, we have that t is the time, y is the solution, h is the step size, and f(t, y) is the derivative of y. 

## Training of the model
The loss function in this code use a L1 norm. The optimization algorithm selected to optimze the loss is Adam.

## Example of ordinary differential equation
Here are three examples of predicted residual of Euler forward together with the targeted residual when solving ordinary differential eqations:

1. Linear ODE

![LinearODE](https://github.com/WilliamN-50/Kex/blob/main/figure/LinearODE/Linear_residual_model2.png)
|:--:| 
|Figure 1: The predicted residual and targeted residual using model 2 for LinearODE.|

2. Van der Pol equation

![VanderPol](https://github.com/WilliamN-50/Kex/blob/main/figure/VanderPol/Vander_residual_model2.png)
|:--:| 
|Figure 2: The predicted residual and targeted residual using model 2 for Van der Pol ODE.|

3. Kepler equation

![Kepler](https://github.com/WilliamN-50/Kex/blob/main/figure/Kepler/Kepler_residual_model2.png)
|:--:| 
|Figure 3: The predicted residual and targeted residual using model 2 for Kepler ODE.|
