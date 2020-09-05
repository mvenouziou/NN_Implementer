"""
This file includes:
    -   Common functions and derivatives used in Gradient Descent methods
        such as activation, cost and regularization functions.
    -   functions_dictionary() to organize and call functions
    -   activation_function() to implement the functions

UPDATES WANTED:
    -   2nd order derivatives
    -   More functions that may be useful for GD
    -   LP for P !=2
"""


import numpy as np


def activation_function(name='sigmoid', derivative=False, x=.5*np.ones((2, 2)), y=1, p=2, s=.1):
    """
    Applies an activation function or its derivative with the given inputs.

    Arguments:
    name: function choice from functions_dictionary
    derivative: bool expression to select if the named function or its derivative should be applied
    x, y, p, s: parameters passed on to named function

    Returns: output of named function
    """

    # Load dictionary of available functions

    activation_functions, cost_functions, regularization_functions, d_functions \
        = functions_dictionary()

    # Update Needed. LP functions not yet implemented for p != 2
    p = 2

    if derivative is False:

        if name in list(regularization_functions):
            func = regularization_functions[name](x, y, p, s)
            return func

        elif name in list(cost_functions):
            func = cost_functions[name](x, y, p, s)
            return func

        elif name in list(activation_functions):
            func = activation_functions[name](x, y, p, s)
            return func

    else:  # derivative is True:

        d_func = d_functions[name](x, y, p, s)
        return d_func
    # End of activation_function() code  ####################################


def functions_dictionary():
    """
    Loads function definitions for use in external functions.

    Returns:
    Dictionary of the form {'function_name': function}. Functions appear without the "()"
    so they don't get computed when functions_dictionary() is called.
    """

    regularization_functions = {'L2 regularization', L2_reg,
                                # 'LP regularization', LP_reg,
                                }

    cost_functions = {'log cost': log_cost,
                      # 'LP cost': LP_cost,
                      'L2 cost': L2_cost,
                      'linear regression': identity,
                      }

    activation_functions = {'sigmoid': sigmoid,
                            'relu': relu,
                            'tanh': tanh,
                            'leaky relu': leaky_relu,
                            'swish': swish,
                            'sin': sin,
                            'identity': identity,
                            # 'softmax': softmax,
                            }

    d_functions = {'log cost': d_log_cost,
                   'sigmoid': d_sigmoid,
                   'relu': d_relu,
                   'tanh': d_tanh,
                   'leaky relu': d_leaky_relu,
                   'swish': d_swish,
                   'sin': d_sin,
                   'identity': d_identity,
                   'LP regularization': d_LP_reg,
                   'LP cost': d_LP_cost,
                   'L2 cost': d_L2_cost,
                   # 'softmax': d_softmax
                   }

    return activation_functions, cost_functions, regularization_functions, d_functions


# FUNCTION DEFINITIONS
# Note: unused variables are needed for implementation to go smoothly in activation_function() above

# REGULARIZATION FUNCTIONS

def LP_reg(x, y, p=2, *args):
    reg = np.sum(x**p)
    return reg


def d_LP_reg(x, y, p=2, *args):
    derivative = p * x**(p-1)
    return derivative


def L2_reg(x, y, *args):
    reg = LP_reg(x, y, 2)
    return reg


def d_L2_reg(x, y, *args):
    derivative = d_LP_reg(x, y, 2)
    return derivative


# COST FUNCTIONS

def log_cost(x, y, *args):
    m = x.shape[1]

    # adjust x to prevent log(0)
    np.where(x == 0, x + 10e-5, x)
    np.where(x == 1, x - 10e-5, x)

    cost = (-1 / m) * (np.dot(y, np.log(x).T) + np.dot(1 - y, np.log(1 - x).T))
    return cost


def d_log_cost(x, y, *args):  # derivative with respect to x
    m = x.shape[1]
    derivative = (-1 / m) * (np.divide(y, x) - np.divide(1 - y, 1 - x))
    return derivative


def LP_cost(x, y, p, *args):
    m = x.shape[1]
    cost = (1/m)*(np.dot(abs(x-y)**(p-1), abs(x-y).T)**(1/p))
    return cost


def d_LP_cost(x, y, p, *args):  # derivative with respect to x
    m = x.shape[1]
    derivative = (1/m)*((x-y)*abs(x-y)**(p-2)) * LP_cost(x, y, p)**(1-p)
    return derivative


def L2_cost(x, y, *args):
    cost = LP_cost(x, y, 2)
    return cost


def d_L2_cost(x, y, *args):  # derivative with respect to x
    derivative = d_LP_cost(x, y, 2)
    return derivative


# OUTPUT LAYER ACTIVATIONS. (z = 1 x m row vector)

def sigmoid(x, *args):
    a = np.divide(1, (1 + np.exp(-x)))
    return a


def d_sigmoid(x, *args):
    derivative = sigmoid(x)*(1-sigmoid(x))
    return derivative


# softmax function
##


# HIDDEN LAYER ACTIVATIONS

def tanh(x, *args):
    a = 2 * sigmoid(2 * x) - 1
    return a


def d_tanh(x, *args):
    derivative = 4 * d_sigmoid(2 * x)
    return derivative


def relu(x, *args):
    a = np.maximum(0, x)
    return a


def d_relu(x, *args):
    derivative = np.where(x >= 0, 1, 0)
    return derivative


def leaky_relu(x, y, p, s):
    a = np.where(x >= 0, x, s*x)
    return a


def d_leaky_relu(x, y, p, s):
    derivative = np.where(x >= 0, 1, s)
    return derivative


def swish(x, *args):
    a = x * sigmoid(x)
    return a


def d_swish(x, *args):
    derivative = sigmoid(x) + x * d_sigmoid(x)
    return derivative


def sin(x, *args):
    # approx_pi = 3.14159265
    a = np.sin(x)
    return a


def d_sin(x, *args):
    # approx_pi = 3.14159265
    derivative = np.cos(x)
    return derivative


def identity(x, *args):  # primarily used in linear regression
    return x


def d_identity(*args):
    derivative = 1
    return derivative

# UNCATEGORIZED


def complex_exp(x, *args):
    a = np.exp(1j*x)
    return a


def d_complex_exp(x, *args):
    deriv = 1j*np.exp(1j*x)
    return deriv
