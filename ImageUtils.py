import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.

        image = np.pad(image,((4,4),(4,4),(0,0)))

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        random_upper_left_x = np.random.randint(9)
        random_upper_left_y = np.random.randint(9)
        image = image[random_upper_left_x:random_upper_left_x+32,random_upper_left_y:random_upper_left_y+32]

        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        flip_or_not = np.random.randint(2)
        if flip_or_not == 1:
            image = np.fliplr(image)
        ### YOUR CODE HERE
        flip_or_not = np.random.randint(2)
        if flip_or_not == 1:
            image = np.flipud(image)

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    std = np.std(image)
    mean = np.mean(image)
    image = (image - mean)/std

    ### YOUR CODE HERE

    return image