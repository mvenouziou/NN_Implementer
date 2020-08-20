"""
This program implements a Neural Network with user-defined size and activation functions

UPDATES WANTED:
    -   regularization / normalize weights or grads - based on number of features/ matrix size?
    -   vary step size or iterations - based on second derivatives? Vary step size by what iteration we are on?
    -   Allow importing multiple folders of images for training set
    -   initialization options for W and b
    -   gradient checking
    -   choice of P for LP cost and regularization. (L2 already implemented)
    -   Utilize complex value weights (with final output activation being the real part of the answer)?
    -   Mini batch implementation
"""


import numpy as np
import pandas as pd
import NN_activation_functions as NN
import NN_data_load_and_shape as DL
import matplotlib.pyplot as plt

# TEST_MODE sets seed value for random numbers and reduces computation time to minimum.
# Use only for program debugging purposes
PRINT_COST = True
TEST_MODE = False

# Test mode - defaults lowered for quicker computer processing
if TEST_MODE is True:
    NUM_ITERATIONS_MODEL = 1000
    PIXELS = (75, 75)
    # np.random.seed(1)
else:
    NUM_ITERATIONS_MODEL = 2500
    PIXELS = (100, 100)

# Hyper-Parameters - Default Values
LEARNING_RATE = 0.0075
HIDDEN_LAYER_SIZES = {"layer_1": 7}

# Activation & Cost - Default Functions
# See NN_activation_functions.py for available options
# Make sure to set a function for each hidden layer
ACTIVATION_FUNCTIONS = {'cost': 'log cost',
                        'layer_1': 'relu',
                        'output_layer': 'sigmoid'}

# Data Sets - Default Values
TRAIN_DATA_FILE = 'datasets/train_catvnoncat.h5'
TEST_DATA_FILE = 'datasets/test_catvnoncat.h5'
FILE_PATH = '/datasets'
DATA_SPLIT_PROP = .8


def NN_creator():
    # Load default values
    activation_functions = ACTIVATION_FUNCTIONS
    layer_sizes = HIDDEN_LAYER_SIZES  # code below will add input & output layer sizes
    learn_rate = LEARNING_RATE
    number_of_iterations = NUM_ITERATIONS_MODEL

    # Load model's functions definitions (as dictionaries)
    activation_functions_choices, cost_function_choices, regularization_functions_choices, d_functions_choices \
        = NN.functions_dictionary()

    # Model customization options:
    # Select training / testing data files
    filetype = input("Is data file h5 type or folder of images? (Enter 'h5' or 'folder'): ")

    if filetype == 'folder':
        pos_data_filename = '.\datasets' + input("Enter folder path for positive data." +
                                                 "Write in format ' \subfolder\* ' : ")

        neg_data_filename = '.\datasets' + input("Enter folder path for negative data." +
                                                 "Write in format ' \subfolder\* ' : ")

        data_split_prop = float(input("Enter data percentage to use for training. "
                                "(Enter as decimal): "))

        image_size = PIXELS

        train_data_file_name = pos_data_filename
        test_data_file_name = pos_data_filename

    else:
        train_data_file_name = input("Enter training data file (or enter for default dataset): ")
        if train_data_file_name == "":
            train_data_file_name = TRAIN_DATA_FILE
        test_data_file_name = input("Enter testing data file (or enter for default dataset): ")
        if test_data_file_name == "":
            test_data_file_name = TEST_DATA_FILE
        data_split_prop = DATA_SPLIT_PROP
        image_size = PIXELS

        pos_data_filename = train_data_file_name
        neg_data_filename = train_data_file_name

    # Load testing and training data
    train_x, test_x, train_y, test_y \
        = DL.prepare_data(file_type = filetype,
                          pos_data_file = pos_data_filename,
                          neg_data_file = neg_data_filename,
                          train_data_file = train_data_file_name,
                          test_data_file = test_data_file_name,
                          size = image_size,
                          proportion = data_split_prop)

    # Select cost functions
    print("\nAvailable Cost Functions:\n", list(cost_function_choices), "\n"
          + "-------------------------------------------------------------", sep='')

    cost_function = 'init'  # initialize prompt loop
    while cost_function not in list(cost_function_choices) + [""]:
        cost_function = input("Enter desired Cost Function or switch to linear regression.\n"
                              + "(Leave blank for default Neural Network Model): ")

    # Update cost function or switch to linear regression
    if cost_function == "":
        skip = True  # skips further customization and uses program defaults

    elif cost_function == 'linear regression':  # switches to standard linear regression
        layer_sizes = {}  # no hidden layers utilized in linear regression
        activation_functions = {'cost': 'L2 cost',
                                'output_layer': 'linear regression'}
        skip = True  # skips further customization and uses linear regression

    else:
        activation_functions['cost'] = cost_function
        skip = False  # allows user to customize model

    # Model Customizations
    if skip is False:

        """ ## REGULARIZATION NEEDS IMPLEMENTATION ##
        ## Select regularization functions
            print("\nAvailable Regularization Functions:", list(regularization_functions_choices), "\n" \
              + "-------------------------------------------------------------", sep='')
    
            regularization_function = 'init'  # initialize prompt loop
            while regularization_function not in list(regularization_functions_choices)+[""]:
                regularization_function = input("Choose regularization or leave blank for program default: ")
        """

        # Select output layer activation function
        print("\nAvailable Activation Functions:\n", list(activation_functions_choices), "\n"
              + "-------------------------------------------------------------", sep='')

        output_activation = 'init'  # initialize prompt loop
        while output_activation not in list(activation_functions_choices) + [""]:
            output_activation = input("Enter output layer activation function.\n"
                                      + "(Leave blank for program default): ")

        if output_activation != "":
            activation_functions['output_layer'] = output_activation

        # Select hidden layer activation functions
        print("\nConfiguring hidden layers.")

        # initiate loop to get hidden layer sizes and activation functions
        layer_num = 0
        size = input("Enter size of hidden layer 1.\n"
                     + "(Leave blank for default hidden layer options): ")

        while size != "":
            layer_num += 1
            size = int(size)

            layer_function = ""
            while layer_function == "":
                layer_function = input("Enter activation function: ")

            # update activations dictionary
            activation_functions['layer_' + str(layer_num)] = layer_function
            layer_sizes['layer_' + str(layer_num)] = size

            # get next layer info and advance layer number
            size = input("\nEnter size of hidden layer " + str(layer_num+1)
                         + " (or press enter if finished with hidden layers): ")

        # User sets learning rate
        rate = input("\nEnter learning rate.\n"
                     + "(Leave blank for program default): ")
        if rate != "":
            learn_rate = float(rate)

        # User sets num of iterations
        iterations = input("\nEnter number of iterations.\n"
                           + "(Leave blank for program default): ")
        if iterations != "":
            number_of_iterations = int(iterations)

    # Update layers dict
    n_x = train_x.shape[0]  # number of features
    layer_sizes['layer_0'] = n_x
    n_y = test_y.shape[0]  # output layer size

    temp = len(layer_sizes)  # this length changes in the next step
    layer_sizes['layer_'+str(temp)] = n_y   # note: number of layers = len(layer_sizes)-1

    # Update activation functions
    activation_functions['layer_' + str(temp)] = activation_functions['output_layer']

    # Display model summary
    print()
    print("Model Structure:")
    for index in range(1, len(layer_sizes)):
        print("Layer:", index, "   Size:", layer_sizes['layer_'+str(index)],
              "   Activation Function:", activation_functions['layer_'+str(index)])
    print("Cost function:", activation_functions['cost'])
    print("Learning Rate:", learn_rate)
    print("Num Iterations:", number_of_iterations)
    print("Training Data File: '", train_data_file_name, "'", sep="")
    print()

    # Create Model consisting of:
    # parameters -- a dictionary containing learned values of W1, W2,... and b1, b2,...
    # cache -- a dictionary containing A0, A1..., Z1, Z2,...
    # grads - - dictionary containing gradient vectors dA1, dA2,...  dZ1, dZ2,...  dW1, dW2,...,  db1, db2,...

    print("Creating model...")
    parameters, cache, grads = \
        k_layer_model(X = train_x, Y = train_y, layers_dims = layer_sizes, functions = activation_functions,
                      learning_rate = learn_rate, num_iterations = number_of_iterations, print_cost=PRINT_COST,
                      test_mode=TEST_MODE)

    # Determine accuracy vs test data
    # UPDATE NEEDED. Currently loads only for sigmoid output activation
    # Works for binary predictions of 0 or 1, where final activation function maps to the interval [0,1]

    last_layer = len(layer_sizes)-1
    if activation_functions['layer_'+str(last_layer)] == 'sigmoid':
        # Load test data into cache and create predictions
        print()
        print("---------- Accuracy on Test Data ----------")

        testing_cache = cache.copy()
        testing_cache['A0'] = test_x

        # Run test data through the model
        test_results_cache = \
            forward_propagation(testing_cache, activation_functions, parameters, number_of_layers = len(layer_sizes)-1)

        # Extract predictions (now as probabilities)
        test_probabilities = test_results_cache['A' + str(len(layer_sizes)-1)]
        test_cost = \
            NN.activation_function(name=activation_functions['cost'], derivative=False,
                                   x = test_probabilities, y= test_y)
        np.squeeze(test_cost)

        # proportion of positive samples in training set. used to adjust criteria for predicting 0 or 1
        prop_train = np.sum(train_y) / len(train_y)

        # Convert model probabilities into predictions of 0 or 1
        test_predictions = np.apply_along_axis(lambda x: np.where(x > prop_train, 1, 0), 0, test_probabilities)

        test_errors = test_y - test_predictions  # nonzero entries indicate incorrect predictions

        number_correct = test_errors.shape[1] - np.count_nonzero(test_errors[0])
        test_accuracy = number_correct/test_errors.shape[1]

        print("Error (cost function):", test_cost)
        print("Percent of correct predictions: ", format(100*test_accuracy, '.2f'), '%', sep = '')
        print(number_correct, "correct predictions out of", test_errors.shape[1])

    # Save Model to File
    # Choose file name
    print("Model parameters will be saved to 'saved_models.h5'.")
    model_name = input("Enter model name: ")
    model_id = 'm' + str(np.random.randint(1, 10**10, dtype=np.int64))
    # note: the 'm' is needed to put in format so 'pd.DataFrame.from_dict' and '.to_hdf' don't trigger warnings

    # Arrange model data friendlier display
    # Alphabetize activation_functions
    tempdict = {}
    for key in sorted(activation_functions):
        tempdict[key] = activation_functions[key]
    activation_functions = tempdict.copy()

    # Alphabetize layer_sizes
    tempdict = {}
    for key in sorted(layer_sizes):
        tempdict[key] = layer_sizes[key]
    layer_sizes = tempdict.copy()

    # Complete model description
    model_info = {'Model Name': model_name,
                  'Model ID': model_id,
                  'Learning Rate': learn_rate,
                  'Layer Sizes': layer_sizes,
                  'Activation Functions': activation_functions,
                  'Parameters': parameters,
                  'Training Data File': train_data_file_name,
                  'Num input features': n_x,
                  'Num output features': n_y,
                  'Gradients': grads,
                  'Image Size': image_size,
                  }

    # Convert to HDF format for saving
    model_dataframe = pd.DataFrame.from_dict(model_info, orient = 'index', columns = [model_info['Model ID']])
    model_dataframe.to_hdf('saved_models.h5', key= str(model_id), mode='a')

    # Write data to file
    model_names_file = open('saved_model_names.txt', 'a')

    for key in ['Model ID', 'Model Name', 'Training Data File', 'Learning Rate',
                'Layer Sizes', 'Activation Functions']:
        model_names_file.write(key + ": " + str(model_info[key]) + "\n")
    model_names_file.write("\n")

    model_names_file.close()

    print("Model #", model_info['Model ID'], "created.")
    # End of NN_creator() code  ####################################


def forward_propagation(caches, func, param, number_of_layers):
    """
    Applies model. A0 = input --> Z1 = W1*A0 + b1 --> A1 = f(Z1) --> ... --> AN

    Arguments:
    caches: a dictionary input vector A0. Access using "caches['A0']"
    func: a dictionary of activation function names for each layer
    param: a dictionary containing W1, W2... and b1, b2...
    number_of_layers

    Returns:
    caches: a dictionary of A0, A1, A2,... and Z1, Z2,...
    """

    for k in range(1, number_of_layers + 1):
        caches['Z' + str(k)] = np.dot(param['W' + str(k)], caches['A' + str(k - 1)]) + param['b' + str(k)]
        caches['A' + str(k)] = NN.activation_function(
                                    name = func['layer_' + str(k)], derivative= False, x = caches['Z' + str(k)])
    return caches
    # End of forward_propagation() code  ####################################


def k_layer_model(X, Y, layers_dims=HIDDEN_LAYER_SIZES, functions=ACTIVATION_FUNCTIONS, learning_rate=LEARNING_RATE,
                  num_iterations=NUM_ITERATIONS_MODEL, print_cost=False, test_mode = False):

    """
    Implements a k-layer neural network.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector corresponding to X
    activation_functions -- ## (cost func, layer1 func, layer2 func, ... etc). Choose from: relu, sigmoid, tanh
    layers_dims -- dimensions of the layers (n_x, n_h1, n_h2... , n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations
    test_mode -- sets seed value for np.random

    Returns:
    parameters -- a dictionary containing learned values of W1, W2,... and b1, b2,...
    cache -- a dictionary containing A0, A1..., Z1, Z2,...
    grads - - dictionary containing gradient vectors dA1, dA2,...  dZ1, dZ2,...  dW1, dW2,...,  db1, db2,...
    """

    costs = []  # to keep track of the cost
    # Note: number of samples = X.shape[1]
    num_layers = len(layers_dims)-1  # note: n_x = layers_dims['layer_0'],  n_y = layers_dims['layer_'+ str(num_layers)]

    # Initialize dictionaries to store parameters, gradients and cache
    parameters = {}
    grads = {}
    cache = {'A0': X,
             'Z0': 1
             }

    # Initialize W parameters with random numbers, b = 0
    for index in range(1, num_layers + 1):
        parameters['W'+str(index)] = np.random.randn(layers_dims['layer_'+str(index)],
                                                     layers_dims['layer_'+str(index-1)]) * .01
        parameters['b'+str(index)] = np.zeros((layers_dims['layer_'+str(index)], 1))

    # Loop to find optimal parameters
    for i in range(0, num_iterations):

        # FORWARD PROPAGATION: LINEAR -> activation -> LINEAR -> activation ###
        cache = forward_propagation(caches = cache, func = functions,
                                    param = parameters, number_of_layers= num_layers)
        # Note: Cache is a dict of Z  = cache['Z' + str(k)] and A = cache['A' + str(k)]
        # for each layer number 'k'

        # Calculate cost
        cost = NN.activation_function(name = functions['cost'], derivative = False,
                                      x = cache['A'+str(num_layers)], y = Y)
        cost = np.squeeze(cost)

        # GRADIENT DESCENT / BACK PROPAGATION
        # Note: all gradients below are of the Cost function. i.e. dF := dC/dF

        # Initialize first value for gradient descent loop
        grads['dA'+str(num_layers)] = NN.activation_function(name =functions['cost'], derivative = True,
                                                             x = cache['A'+str(num_layers)], y = Y)

        for k in range(num_layers, 0, -1):
            grads['dZ' + str(k)] = grads['dA' + str(k)] * NN.activation_function(name=functions['layer_'+str(k)],
                                                                                 derivative= True, x=cache['Z'+str(k)])
            grads['dW' + str(k)] = np.dot(grads['dZ'+str(k)], cache['A'+str(k-1)].T)
            grads['db' + str(k)] = np.dot(grads['dZ'+str(k)], np.ones((grads['dZ'+str(k)].shape[1], 1)))
            grads['dA' + str(k-1)] = np.dot(parameters['W'+str(k)].T, grads['dZ'+str(k)])

        # Update parameters.
        for k in range(1, num_layers+1):
            parameters['W'+str(k)] = parameters['W'+str(k)] - learning_rate * grads['dW'+str(k)]
            parameters['b'+str(k)] = parameters['b'+str(k)] - learning_rate * grads['db'+str(k)]

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot costs
    if test_mode is True:

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters, cache, grads
    # End of k_layer_model() code  ####################################
