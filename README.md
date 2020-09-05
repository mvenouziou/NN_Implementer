# NN_Implementer

NN_implementer is meant as a way to implement user-customized Neural Network models. It is a side project where I add features over time while I take courses in Machine Learning.
COMPONENTS:

NN_model_implementation.py: Interface to accept user's data and train custom Neural Networks.

NN_with_gradient_descent.py: Implements Gradient Descent training and testing. Accepts user inputs of layer sizes, depth and activation functions.

NN_activation_functions.py: Common loss, activation and regularization functions and their derivatives. Functions defined here automatically integrate and appear as an options for the Gradient Descent algorithm.

NN_data_load_and_shape.py: helper functions to accept training data and format it as needed by our GD algorithm
