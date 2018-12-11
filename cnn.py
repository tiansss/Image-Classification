import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import math
import random
import glob
import cv2
from sklearn.utils import shuffle
from imutils import paths

train_path = 'dataset'
checkpoint_dir = 'ckpoint'
learning_rate = 0.0001
MIN_CATEGORY_SIZE = 200

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 64

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
origin_classes = [name for name in os.listdir(train_path)]
classes = []
# Read class info, edit this part to get different numbers of categories
for fld in origin_classes:
    path = os.path.join(train_path, fld, '*g')
    files = glob.glob(path)
    if (len(files) >= MIN_CATEGORY_SIZE):
        classes.append(fld)
classes_low = random.sample(classes, 10)
# classes = []
# for fld in origin_classes:
#     path = os.path.join(train_path, fld, '*g')
#     files = glob.glob(path)
#     if (len(files) >= 3 * MIN_CATEGORY_SIZE):
#         classes.append(fld)
# classes_high = random.sample(classes, 5)
classes = classes_low
# classes.extend(classes_high)
num_classes = len(classes)

# batch size
batch_size = 32

# train split
train_size = .7
test_size = .5 # the proportion of test data on the data except for train data

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

conv_layers = [
    {
        'f_size': 3,
        'num_outputs': 64,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 64,
        'use_pool': True,
    },
    {
        'f_size': 3,
        'num_outputs': 64,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 64,
        'use_pool': True,
    },
    {
        'f_size': 3,
        'num_outputs': 128,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 128,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 128,
        'use_pool': True,
    },
    {
        'f_size': 3,
        'num_outputs': 256,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 256,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 256,
        'use_pool': True,
    },
    {
        'f_size': 3,
        'num_outputs': 512,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 512,
        'use_pool': False,
    },
    {
        'f_size': 3,
        'num_outputs': 512,
        'use_pool': True,
    },
]

fc_layers = [
    {
        'num_outputs': 2048,
        'keep_prob': 0.5
    },
    {
        'num_outputs': 2048,
        'keep_prob': 0.5
    },
    {
        'num_outputs': num_classes,
        'keep_prob': 1
    }
]

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []
    stratify = []

    print('Reading training images')
    for fld in classes:
        index = classes.index(fld)
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        print('Loading {} files (Index: {})'.format(fld, index))
        for fl in files:
            # if files.index(fl) > 99:
            #     break
            stratify.append(index)
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    stratify = np.array(stratify)

    return images, labels, ids, cls, stratify

class DataSet(object):

    def __init__(self, images, labels, ids, cls):
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""

        self._num_examples = images.shape[0]


        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # Convert from [0, 255] -> [0.0, 1.0].

        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, train_size, test_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, ids, cls, stratify = load_train(train_path, image_size, classes)

    train_images, images, train_labels, labels, train_ids, ids, train_cls, cls, train_stratify, stratify =\
        train_test_split(images, labels, ids, cls, stratify, train_size=train_size, test_size=None, stratify=stratify)
    validation_images, test_images, validation_labels, test_labels, validation_ids, test_ids, validation_cls, test_cls =\
        train_test_split(images, labels, ids, cls, test_size=test_size, stratify=stratify)

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)
    data_sets.test = DataSet(test_images, test_labels, test_ids, test_cls)

    return data_sets

# load training dataset
data = read_train_sets(train_path, img_size, classes, train_size, test_size)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation:\t{}".format(len(data.valid.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))

# Helper-functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# Convolutional Layer
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_outputs,        # Number of filters.
                   use_pool):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_outputs]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_outputs)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pool:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer

# Flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

# Fully-Connected Layer
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 keep_prob):     # Keep probability.

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.relu(layer)

    # Dropout layer.
    if keep_prob < 1:
        layer = tf.nn.dropout(layer, keep_prob)

    return layer

# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Convolutional Layers
def build_conv_layers():
    conv_layer = new_conv_layer(x_image, num_channels, conv_layers[0]['f_size'], conv_layers[0]['num_outputs'], False)
    for i in range(1, len(conv_layers)):
        conv_layer = new_conv_layer(conv_layer, conv_layers[i - 1]['num_outputs'], conv_layers[i]['f_size'], conv_layers[i]['num_outputs'], conv_layers[i]['use_pool'])
    return conv_layer

last_conv_layer = build_conv_layers()

# Flatten Layer
flat_layer, num_features = flatten_layer(last_conv_layer)
print(flat_layer, num_features)

# Fully-Connected Layers
def build_fc_layers():
    fc_layer = new_fc_layer(flat_layer, num_features, fc_layers[0]['num_outputs'], fc_layers[0]['keep_prob'])
    for i in range(1, len(fc_layers)):
        fc_layer = new_fc_layer(fc_layer, fc_layers[i - 1]['num_outputs'], fc_layers[i]['num_outputs'], fc_layers[i]['keep_prob'])
    return fc_layer

last_fc_layer = build_fc_layers()

# Predicted Class
y_pred = tf.nn.softmax(last_fc_layer)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_fc_layer,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = batch_size

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=8000)

x_test = data.test.images.reshape(len(data.test.labels), img_size_flat)
feed_dict_test = {x: x_test, y_true: data.test.labels}
val_loss = session.run(cost, feed_dict=feed_dict_test)
val_acc = session.run(accuracy, feed_dict=feed_dict_test)
msg_test = "Test Accuracy: {0:>6.1%}"
print(msg_test.format(val_acc))
