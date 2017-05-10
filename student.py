"""
                     PA5 STUDENT IMPLEMENATION
                     -------------------------
"""

# Answer TODO 1 as a comment here:
############################ TODO 1 BEGIN #################################
#
# Whereas fc6 and fc7 are transformed by both ReLU and dropout in-place layers
# (meaning that they are directly transformed), fc8 is simply
# output as is. A ReLU layer will transform the output to be non-negative, so the
# lack of one attached to fc8 lets it span both negative and positive output values.
#
############################ TODO 1 END #################################


# Add imports here
import numpy as np


def convert_ilsvrc2012_probs_to_dog_vs_food_probs(probs_ilsvrc):
    """
    Convert from 1000-class ILSVRC probabilities to 2-class "dog vs food"
    incices.  Use the variables "dog_indices" and "food_indices" to map from
    ILSVRC2012 classes to our classes.

    HINT:
    Compute "probs" by first estimating the probability of classes 0 and 1,
    using probs_ilsvrc.  Stack together the two probabilities along axis 1, and
    then normalize (along axis 1).

    :param probs_ilsvrc: shape (N, 1000) probabilities across 1000 ILSVRC classes

    :return probs: shape (N, 2): probabilities of each of the N items as being
        either dog (class 0) or food (class 1).
    """
    # in the ILSVRC2012 dataset, indices 151-268 are dogs and index 924-969 are foods
    dog_indices = range(151, 269)
    food_indices = range(924, 970)
    N, _ = probs_ilsvrc.shape
    probs = np.zeros((N, 2)) # placeholder

    ############################ TODO 2 BEGIN #################################
    dogSum = 0
    foodSum = 0

    # for every image in the test set
    for i in range(N):
        # calculate the sums at the correct indexes
        dogSum = sum(probs_ilsvrc[i, dog_indices])
        foodSum = sum(probs_ilsvrc[i, food_indices])

        # put the probabilities in place and normalize
        probs[i] = (dogSum, foodSum)
        probs[i] = probs[i]/sum(probs[i])

    ############################ TODO 2 END #################################
    return probs


def get_prediction_descending_order_indices(probs, cidx):
    """
    Returns the ordering of probs that would sort it in descending order

    :param probs: (N, 2) probabilities (computed in TODO 2)
    :param cidx: class index (0 or 1)

    :return list of N indices that sorts the array in descending order
    """
    order = range(probs.shape[0]) # placeholder
    ############################ TODO 3 BEGIN #################################

    # perform the sort (automatically returns in ascending order, so need to reverse with [::-1])
    order = np.argsort(probs[:,cidx])[::-1]

    ############################ TODO 3 END #################################
    return order


def compute_dscore_dimage(net, data, class_idx):
    """
    Returns the gradient of s_y (the score at index class_idx) with respect to
    the image (data), ds_y / dI.  Note that this is the unnormalized class
    score "s", not the probability "p".

    :param data: (3, 227, 227) array, input image
    :param class_idx: class index in range [0, 999] indicating which class
    :param net: a caffe Net object

    :return grad: (3, 227, 227) array, gradient ds_y / dI
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 4 BEGIN #################################

    # set fc8 diff layer to 0 except at class_idx, where = 1
    net.blobs['fc8'].diff[0, ...] = 0
    net.blobs['fc8'].diff[0, class_idx] = 1

    # back propagate the network
    net.backward(start='fc8', end='data')

    # get the diff of the image data layer
    grad = np.copy(net.blobs['data'].diff[0])

    ############################ TODO 4 END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


def normalized_sgd_with_momentum_update(data, grad, velocity, momentum, learning_rate):
    """
    THIS IS SLIGHTLY DIFFERENT FROM NORMAL SGD+MOMENTUM; READ THE NOTEBOOK :)

    Update the image using normalized SGD+Momentum.  To make learning more
    stable, normalize the gradient before using it in the update rule.

    :param data: shape (3, 227, 227) the current solution
    :param grad: gradient of tthe loss with respect to the image
    :param velocity: momentum vector "V"
    :param momentum: momentum parameter "mu"
    :param learning_rate: learning rate "alpha"

    :return: the updated image and momentum vector (data, velocity)
    """
    ############################ TODO 5a BEGIN #################################
    raise NotImplementedError("TODO 5a")
    ############################ TODO 5a BEGIN #################################
    return data, velocity


def fooling_image_gradient(net, orig_data, data, target_class, regularization):
    """
    Compute the gradient for make_fooling_image (dL / dI).

    :param net: a caffe Net object
    :param orig_data: shape (3, 227, 227) the original image
    :param target_class: ILSVRC class in range [0, 999]
    :param data: shape (3, 227, 227) the current solution
    :param regularization: weight (lambda) applied to the regularizer.
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 5b BEGIN #################################
    raise NotImplementedError("TODO 5b")
    ############################ TODO 5b END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


def class_visualization_gradient(net, data, target_class, regularization):
    """
    Compute the gradient for make_class_visualization (dL / dI).

    :param net: a caffe Net object
    :param target_class: ILSVRC class in range [0, 999]
    :param data: shape (3, 227, 227) the current solution
    :param regularization: weight (lambda) applied to the regularizer.
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 6 BEGIN #################################
    raise NotImplementedError("TODO 6")
    ############################ TODO 6 END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


def feature_inversion_gradient(net, data, blob_name, target_feat, regularization):
    """
    Compute the gradient for make_feature_inversion (dL / dI).

    :param net: a caffe Net object
    :param data: shape (3, 227, 227) the current solution
    :param blob_name: which caffe blob name (script \ell in the notebook)
    :param target_feat: target feature
    :param regularization: weight (lambda) applied to the regularizer.
    """
    grad = np.zeros_like(data) # placeholder
    ############################ TODO 7a BEGIN #################################
    raise NotImplementedError("TODO 7a")
    ############################ TODO 7a END #################################
    assert grad.shape == (3, 227, 227) # expected shape
    return grad


# Answer TODO 7b as a comment here:
############################ TODO 7b BEGIN #################################
#
# (a)
#
#
#
# (b)
#
#
#
############################ TODO 7b END #################################
