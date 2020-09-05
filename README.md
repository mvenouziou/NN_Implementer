# NN_Implementer

Neural Network modeler and user interface coded from scratch.
Innovations compared to scikit-learn: 
- activation functions can be chosen per layer
- define new activation / cost functions with only minor code updates.
- allow identity function to replace activation function on subset of nodes within each layer.

### COMPONENTS:

#### NN_model_implementation.py: 
Interface to accept user's data and train custom Neural Networks. Models can be saved to file for later use.

#### NN_with_gradient_descent.py: 
Implements Gradient Descent training and testing. Accepts user inputs of layer sizes, depth and activation functions.

#### NN_activation_functions.py: 
Common loss, activation and regularization functions and their derivatives. 
Add new functions by simply defining function and its derivative. (Function must support numpy broadcasting.) 
New functions automatically integrate into NN_with_gradient_descent.

#### NN_data_load_and_shape.py: 
helper functions to accept training data and format it as needed by our GD algorithm
