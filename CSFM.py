# https://github.com/igv/FSRCNN-TensorFlow

import tensorflow as tf

class Model(object):

  def __init__(self, config):
    self.name = "CSFM"
    self.model_params = [8, 2, 4]
    self.scale = config.scale
    self.radius = config.radius
    self.padding = config.padding
    self.images = config.images
    self.labels = config.labels
    self.batch = config.batch
    self.image_size = config.image_size - self.padding
    self.label_size = config.label_size

  def model(self):
    d, m, b = self.model_params

    size = self.padding + 1
    features = tf.contrib.layers.conv2d(self.images, d, size, 1, 'VALID', 'NHWC', activation_fn=None, scope='features')

    conv = tf.contrib.layers.conv2d(features, d, 3, 1, 'SAME', 'NHWC', activation_fn=None, scope='conv1')

    shortcuts = conv

    for i in range(1, m+1):
        with tf.variable_scope("FMM{}".format(i)) as scope:
            for bi in range(1, b+1):
                res = tf.contrib.layers.conv2d(conv, d*6, 1, 1, 'SAME', 'NHWC', activation_fn=tf.nn.leaky_relu, scope='widen{}'.format(bi))
                res = tf.contrib.layers.conv2d(res, d, 1, 1, 'SAME', 'NHWC', activation_fn=None, scope='shrink{}'.format(bi))
                res = tf.contrib.layers.conv2d(res, d, 3, 1, 'SAME', 'NHWC', activation_fn=None, scope='embedding{}'.format(bi))

                sa = tf.contrib.layers.separable_conv2d(res, None, 3, 1, 1, 'SAME', 'NHWC', activation_fn=None, scope='sa{}'.format(bi))

                ca = tf.reduce_mean(tf.square(res), [1, 2], True) - tf.square(tf.reduce_mean(res, [1, 2], True))
                ca = tf.contrib.layers.conv2d(ca, max(d//16, 4), 1, 1, 'SAME', 'NHWC', activation_fn=tf.nn.leaky_relu, scope='ca_shrink{}'.format(bi))
                ca = tf.contrib.layers.conv2d(ca, d, 1, 1, 'SAME', 'NHWC', activation_fn=None, scope='ca{}'.format(bi))

                conv = tf.add(conv, tf.add(res, tf.multiply(res, tf.sigmoid(tf.add(sa, ca)))))

            conv = tf.concat([conv, shortcuts], -1)
            conv = tf.contrib.layers.conv2d(conv, d, 1, 1, 'SAME', 'NHWC', activation_fn=None, scope='GF{}'.format(i))
            shortcuts = tf.concat([conv, shortcuts], -1)

    conv = tf.contrib.layers.conv2d(conv, d, 3, 1, 'SAME', 'NHWC', activation_fn=None, scope='res')
    conv = tf.add(conv, features)

    with tf.variable_scope("upscaling"):
        conv = tf.nn.leaky_relu(conv)
        conv = tf.contrib.layers.conv2d(conv, d * self.scale**2, 3, 1, 'SAME', 'NHWC', activation_fn=None, scope='sub-pixel_conv')
        conv = tf.depth_to_space(conv, self.scale, name='pixel_shuffle', data_format='NHWC')

        conv = tf.contrib.layers.conv2d(conv, 1, 3, 1, 'SAME', 'NHWC', activation_fn=None, scope='final')

    return conv

  def loss(self, Y, X):
    dY = tf.image.sobel_edges(Y)
    dX = tf.image.sobel_edges(X)
    M = tf.sqrt(tf.square(dY[:,:,:,:,0]) + tf.square(dY[:,:,:,:,1]))
    return tf.losses.absolute_difference(dY, dX) \
         + tf.losses.absolute_difference((1.0 - M) * Y, (1.0 - M) * X, weights=2.0)
