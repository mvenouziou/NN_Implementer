"""
This is the user interface for creating and applying Neural Network models

NEEDED FIXES:
-   test model against file not working

DESIRED IMPROVEMENTS:
-   Implement for non-image models
-   predictions on folders of images
"""

import numpy as np
import pandas as pd
from PIL import Image
import NN_with_gradient_descent as GD


CRITERIA = .5  # probability threshold for making predictions with model
DEFAULT_MODEL_ID = 'm8898130196'  # cat image detector
IMAGE_FILE = 'images/bw.jpg'  # sample cat image
SHOW_IMAGE = False

# train_data_file = h5py.File(train_data, "r")
# an_image = train_data_file["train_set_x"][0]
# an_image = Image.open(an_image)


def main():
    train = input("Train model with gradient descent? (y/n): ")

    while train in ['y', 'Y']:
        GD.NN_creator()
        train = input("Train another model? (y/n): ")

    make_predictions = input("Test image against model? (y/n): ")

    if make_predictions in ['y', 'Y']:
        model_id = input("Enter Model ID to use (or press enter to use default model): ")
        if model_id == "":
            model_id = DEFAULT_MODEL_ID

        while make_predictions in ['y', 'Y']:
            predictor(mod_id = model_id)
            make_predictions = input("Test another image? (y/n): ")


def predictor(mod_id = DEFAULT_MODEL_ID):

    # load model and extract dictionaries
    model_dataframe = pd.read_hdf('saved_models.h5', key = mod_id)
    model_dict = model_dataframe.to_dict()[mod_id]

    # model_dict is a dictionary of the form
    """
    model_dict = {'Model Name': model_name,
                  'Model ID': model_id,
                  'Learning Rate': learn_rate,
                  'Layer Sizes': layer_sizes,
                  'Activation Functions': activation_functions,
                  'Parameters': parameters,
                  'Training Data File': train_data_file_name,
                  'Num input features': n_x,
                  'Num output features': n_y,
                  'Gradients': grads,
                  'Image Size': image_size  # of the form (px, py)
                  }
    
    """

    # display model info to user:
    print()
    for key in ['Model ID', 'Model Name', 'Training Data File', 'Learning Rate',
                'Layer Sizes', 'Activation Functions']:
        print(key, ":", model_dict[key])

    # Load file for prediction(s)
    filename = input("\nEnter image file to have model applied to: ")

    # for images
    image_file = Image.open(filename)
    resized_image \
        = image_file.resize(model_dict['Image Size'])

    if SHOW_IMAGE is True:
        resized_image.show()

    # shape and normalize to match model's data
    image_array = np.array(resized_image)
    X = image_array.reshape(-1, 1)
    X = X / np.linalg.norm(X)

    # apply model
    num_layers = len(model_dict['Layer Sizes'])-1

    cache = GD.forward_propagation(caches= {'A0': X},
                                   func= model_dict['Activation Functions'],
                                   param= model_dict['Parameters'],
                                   number_of_layers= num_layers
                                   )

    # model output
    probabilities = cache['A' + str(num_layers)][0]

    # apply prediction criteria and display results
    predictions \
        = np.apply_along_axis(lambda x: np.where(x > .5, 1, 0), 0, probabilities)

    print()
    print("Probabilities: ", probabilities[:10], "...")
    print("Predictions: ", predictions)


main()
