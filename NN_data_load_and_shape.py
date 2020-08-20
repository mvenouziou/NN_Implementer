"""
This file contains programs to load and format data for use in NN_with_gradient_descent.py

    -   prepare_data(): Converts training and testing data files into format required
    by NN_with_gradient_descent(). Accepts .5h files in array form
    or folders of images in PIL-readable format.

    -   import_images_from_directory(): Converts a folder of images into RGB images of given size,
    and produces two arrays, image array and labels array, where columns correspond to flattened
    image vectors. Labels are either 0 or 1.

    -   split_into_test_and_train(): Takes two arrays as inputs, randomly rearranges their column orders
    (same reordering for each array) and then splits each into two arrays.

    -   load_h5_data(): helper function for importing h5 files.

    -   reshape_h_5_data(): helper function for formating h5 files as arrays.


UPDATES WANTED:
    -   Add data import from csv
    -   Allow importing multiple folders of images for training set
"""


import numpy as np
import h5py
import PIL
import PIL.Image
import glob


def prepare_data(file_type='h5', pos_data_file = '/datasets', neg_data_file = '/datsets',
                 train_data_file='datasets/train_catvnoncat.h5', test_data_file='datasets/test_catvnoncat.h5',
                 size = (250, 250), proportion = .8):
    """
    Converts training and testing data files into format required by NN_with_gradient_descent().
    Accepts .5h files in array form or folders with image PIL readable files.

    Arguments:
    train_data_file: filename of training data.
    test_data_file: filename of testing data

    Returns:
    train x, test x: numpy arrays of shape (# features, # samples).
      Each column is a flattened input vector from the training or testing set.
    train_set_y_orig, test_set_y_orig: numpy array of shape (# labels, # samples).
        Each column contains the label(s) corresponding to a sample from the
        training or testing set.

    """

    if file_type == 'h5':
        train_set_x_orig, train_y, test_set_x_orig, test_y, classes \
            = load_h5_data(train_data_file, test_data_file)

        train_x, test_x \
            = reshape_h_5_data(train_set_x_orig, test_set_x_orig)

    else:  # file_type == 'images':
        img_array, labels \
            = import_images_from_directory(pos_data_file, neg_data_file, size)

        train_x, test_x, train_y, test_y,  \
            = split_into_test_and_train(img_array, labels, proportion)

    return train_x, test_x, train_y, test_y


def import_images_from_directory(positive_samples_path, negative_samples_path, size = (250, 250)):
    """
    This converts a folder of images into RGB images of given size,
    converts each image into a column vector and produces two arrays - image array and labels array.
    Labels are 1 for 'positive_samples_path' and 0 for 'negative_samples_path'.

    Arguments:
    path: folder name containing images
    label_name: image label for classification
    size: dimensions (x,y). All images will be shrunk/stretched to this size

    Returns: image array, labels array
    """

    # Load image names
    positive_images_files_list = glob.glob(pathname=positive_samples_path, recursive=True)
    negative_images_files_list = glob.glob(pathname=negative_samples_path, recursive=True)

    # Sample size
    m = len(positive_images_files_list) + len(negative_images_files_list)

    # Create arrays to hold image vectors
    labels = []
    img_array = np.zeros((size[0] * size[1] * 3, m), dtype= np.int)  # images will be converted to RGB (3 colors)

    # loop to add images into array
    col_num = 0  # prepare loop
    for image in positive_images_files_list:

        # Import and format image
        img_data = PIL.Image.open(image).resize(size).convert(mode = 'RGB', colors = 256)

        # Convert to array and flatten
        temp_image_vector = np.array(img_data, dtype=np.int)
        temp_image_vector = temp_image_vector.reshape(size[0] * size[1] * 3)

        # normalize to unit vector
        temp_image_vector \
            = temp_image_vector / np.linalg.norm(temp_image_vector)

        # Replace column = col_num with image vector
        img_array[:, col_num] = temp_image_vector

        # update labels list
        labels += [1]
        col_num += 1

        # Close image file
        img_data.close()

    # loop to add negative images into array
    for image in negative_images_files_list:

        # Import and format image
        img_data = PIL.Image.open(image).resize(size).convert(mode = 'RGB', colors = 256)

        # Convert to array and flatten
        temp_image_vector = np.array(img_data, dtype=np.int)
        temp_image_vector = temp_image_vector.reshape(size[0] * size[1] * 3)

        # normalize to unit vector
        temp_image_vector \
            = temp_image_vector / np.linalg.norm(temp_image_vector)

        # Replace column = col_num with image vector
        img_array[:, col_num] = temp_image_vector

        # update labels list
        labels += [0]
        col_num += 1

        # Close image file
        img_data.close()

    # format labels as array
    labels = np.array(labels).reshape((1, m))

    return img_array, labels


def split_into_test_and_train(orig_array_a, orig_array_b, proportion = .8):

    """
    This function takes, two arrays and randomly rearranges their column order
    then splits each into two arrays.
    The number of columns in each new array is determined by proportion value

    Arguments
    array: original numpy array
    proportion: proportion of the original columns that go to new array 1

    Returns: new_array_a1, new_array_a2, new_array_b1, new_array_b2
    """

    # Number of columns in original array
    m = len(orig_array_a[0])

    # Randomize order of orig_array_a and orig_array_b
    rand_column_order = np.random.choice(m, m, replace=False)
    orig_array_a[:, np.arange(m)] = orig_array_a[:, rand_column_order]
    orig_array_b[:, np.arange(m)] = orig_array_b[:, rand_column_order]

    # Determine number of columns for new_array_1
    num_columns = int(m * proportion)

    # Split arrays
    # Arrays 'a1', 'a2'
    new_arrays_a = np.hsplit(orig_array_a, [num_columns, m])
    new_array_a1 = new_arrays_a[0]
    new_array_a2 = new_arrays_a[1]

    # Arrays 'b1', 'b2'
    new_arrays_b = np.hsplit(orig_array_b, [num_columns, m])
    new_array_b1 = new_arrays_b[0]
    new_array_b2 = new_arrays_b[1]

    return new_array_a1, new_array_a2, new_array_b1, new_array_b2


"""
def load_and_separate_into_test_train(images_array, labels_array, prop = .8):
    images_array, labels_array = \
        import_images_from_directory(positive_samples_path = './datasets/Emmy/*',
                                     negative_samples_path = './datasets/other pets/*', size=(250, 250))

    train_array_image, test_array_image, train_array_labels, test_array_labels \
        = split_into_test_and_train(images_array, labels_array, prop)

    return train_array_image, test_array_image, train_array_labels, test_array_labels
"""


def load_h5_data(train_data_file, test_data_file):

    train_dataset = h5py.File(train_data_file, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(test_data_file, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def reshape_h_5_data(train_x_orig, test_x_orig):
    # Reshape the training and test examples
    train_x_flatten = \
        train_x_orig.reshape(train_x_orig.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = \
        test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print()
    print("- training set shape: " + str(train_x.shape))
    print("- testing set shape: " + str(test_x.shape))

    return train_x, test_x
