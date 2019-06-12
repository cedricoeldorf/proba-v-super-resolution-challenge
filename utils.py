import os
import glob
from math import ceil
import subprocess
import io
from random import randrange, shuffle
from sklearn.externals import joblib
import tensorflow as tf
from PIL import Image
import numpy as np
from multiprocessing import Pool, Lock, active_children
import pickle
FLAGS = tf.app.flags.FLAGS

def single_image_mse(number):
    import joblib
    import cv2
    img1 = joblib.load('./x_test_hr.pickle')[number]
    img2 = joblib.load('./lr.pickle')['x_test_lr'][number]
    img2 = cv2.resize(img2,(384,384), interpolation = cv2.INTER_CUBIC)

    mse = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    mse /= float(img1.shape[0] * img1.shape[1])
    return mse

def train_input_setup(config):
  """
  Read image files, make their sub-images, and save them as a h5 file format.
  """
  sess = config.sess
  image_size, label_size, stride, scale, padding = config.image_size, config.label_size, config.stride, config.scale, config.padding // 2
  sub_input_sequence, sub_label_sequence = [], []

  # Load training images
  with open('lr.pickle', 'rb') as pickle_in:
      lr = pickle.load(pickle_in)
  x_train_hr = joblib.load('x_train_hr.pickle')
  x_train_lr = lr["x_train_lr"]
  del lr

  # Create sub-images
  for i in range(len(x_train_lr)):
    input_, label_ = x_train_lr[i], x_train_hr[i]

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    for x in range(0, h - image_size + 1, stride):
      for y in range(0, w - image_size + 1, stride):
        sub_input = input_[x : x + image_size, y : y + image_size]
        #x_loc, y_loc = x + padding, y + padding
        x_loc, y_loc = x, y
        sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

        sub_input = sub_input.reshape([image_size, image_size, 1])
        sub_label = sub_label.reshape([label_size, label_size, 1])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  return (arrdata, arrlabel)


def test_input_setup(config, number):

  ''' number selects a random image pair from test set '''

  with open('lr.pickle', 'rb') as pickle_in:
    lr = pickle.load(pickle_in)

  x_test_hr = joblib.load('x_test_hr.pickle')
  x_test_lr = lr["x_test_lr"]
  del lr

  input_, label_ = x_test_lr[number], x_test_hr[number]

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  arrdata = input_.reshape([1, h, w, 1])
  if len(label_.shape) == 3:
    h, w, _ = label_.shape
  else:
    h, w = label_.shape

  arrlabel = label_.reshape([1, h, w, 1])

  return (arrdata, arrlabel)

def save_params(sess, params):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  h = open(param_dir + "weights{}.txt".format('_'.join(str(i) for i in params)), 'w')

  variables = dict((var.name, sess.run(var)) for var in tf.trainable_variables())

  for name, weights in variables.items():
    h.write("{} =\n".format(name[:name.index(':')]))

    if len(weights.shape) < 4:
        h.write("{}\n\n".format(weights.flatten().tolist()))
    else:
        h.write("[")
        sep = False
        for filter_x in range(len(weights)):
          for filter_y in range(len(weights[filter_x])):
            filter_weights = weights[filter_x][filter_y]
            for input_channel in range(len(filter_weights)):
              for output_channel in range(len(filter_weights[input_channel])):
                val = filter_weights[input_channel][output_channel]
                if sep:
                    h.write(', ')
                h.write("{}".format(val))
                sep = True
              h.write("\n  ")
        h.write("]\n\n")

  h.close()

def array_image_save(array, image_path):
  """
  Converts np array to image and saves it
  """
  image = Image.fromarray(array, 'YCbCr')
  if image.mode != 'RGB':
    image = image.convert('RGB')
  image.save(image_path)
  print("Saved image: {}".format(image_path))

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, sigma=1.5):
    size = int(sigma * 3) * 2 + 1
    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.abs(tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu1_sq)
    sigma2_sq = tf.abs(tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu2_sq)
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu1_mu2

    if cs_map:
        value = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, sigma=1.5, weights=[0.1, 0.9]):
    weights = weights / np.sum(weights)
    window = _tf_fspecial_gauss(5, 1)
    mssim = []
    for i in range(len(weights)):
        mssim.append(tf_ssim(img1, img2, sigma=sigma))
        img1 = tf.nn.conv2d(img1, window, [1,2,2,1], 'VALID')
        img2 = tf.nn.conv2d(img2, window, [1,2,2,1], 'VALID')

    value = tf.reduce_sum(tf.multiply(tf.stack(mssim), weights))

    return value
