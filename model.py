from utils import (

  train_input_setup,
  test_input_setup,
  save_params,
  array_image_save
)

import time
import os
import importlib
from random import randrange

import numpy as np
import tensorflow as tf

from PIL import Image
import pdb

# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class Model(object):

  def __init__(self, sess, config):
    self.sess = sess
    self.arch = config.arch
    self.fast = config.fast
    self.train = config.train
    self.epoch = config.epoch
    self.scale = config.scale
    self.radius = config.radius
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.threads = config.threads
    self.distort = config.distort
    self.params = config.params

    self.padding = 0
    # Different image/label sub-sizes for different scaling factors x2, x3, x4
    scale_factors = [[20 + self.padding, 40], [14 + self.padding, 42], [12 + self.padding, 48]]
    self.image_size, self.label_size = scale_factors[self.scale - 2]

    self.stride = self.image_size - self.padding

    self.checkpoint_dir = config.checkpoint_dir
    self.output_dir = config.output_dir
    self.data_dir = config.data_dir
    self.init_model()
    #print("CALCULATED IMAGE SIZE ", self.image_size)


  def init_model(self):

    if self.train:
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 1], name='labels')
    else:
        self.images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
        self.labels = tf.placeholder(tf.float32, [None, None, None, 1], name='labels')
    # Batch size differs in training vs testing
    self.batch = tf.placeholder(tf.int32, shape=[], name='batch')

    model = importlib.import_module(self.arch)
    self.model = model.Model(self)

    self.pred = self.model.model()

    model_dir = "%s_%s_%s_%s" % (self.model.name.lower(), self.label_size, '-'.join(str(i) for i in self.model.model_params), "r"+str(self.radius))
    self.model_dir = os.path.join(self.checkpoint_dir, model_dir)

    self.loss = self.model.loss(self.labels, self.pred)

    self.saver = tf.train.Saver()

  def run(self):
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    deconv_mult = lambda grads: list(map(lambda x: (x[0] * 1.0, x[1]) if 'deconv' in x[1].name else x, grads))
    grads = deconv_mult(optimizer.compute_gradients(self.loss))
    self.train_op = optimizer.apply_gradients(grads, global_step=global_step)

    tf.global_variables_initializer().run()

    if self.load():
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")


    if self.params:
      save_params(self.sess, self.model.model_params)
    elif self.train:
      self.run_train()
      self.run_test()
    else:
      self.run_test()

  def run_train(self):
    start_time = time.time()
    print("Beginning training setup...")
    if self.threads == 1:
      train_data, train_label = train_input_setup(self)

    print("Training setup took {} seconds with {} threads".format(time.time() - start_time, self.threads))

    print("Training...")
    start_time = time.time()
    start_average, end_average, counter = 0, 0, 0

    for ep in range(self.epoch):
      # Run by batch images
      batch_idxs = len(train_data) // self.batch_size
      batch_average = 0
      for idx in range(0, batch_idxs):
        batch_images = train_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = train_label[idx * self.batch_size : (idx + 1) * self.batch_size]

        for exp in range(3):
            if exp==0:
                images = batch_images
                labels = batch_labels
            elif exp==1:
                k = randrange(3)+1
                images = np.rot90(batch_images, k, (1,2))
                labels = np.rot90(batch_labels, k, (1,2))
            elif exp==2:
                k = randrange(2)
                images = batch_images[:,::-1] if k==0 else batch_images[:,:,::-1]
                labels = batch_labels[:,::-1] if k==0 else batch_labels[:,:,::-1]
            counter += 1
            _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: images, self.labels: labels, self.batch: self.batch_size})
            batch_average += err

            if counter % 10 == 0:
              print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                % ((ep+1), counter, time.time() - start_time, err))

            # Save every 500 steps
            if counter % 500 == 0:
              self.save(counter)

      batch_average = float(batch_average) / batch_idxs
      if ep < (self.epoch * 0.2):
        start_average += batch_average
      elif ep >= (self.epoch * 0.8):
        end_average += batch_average

    # Compare loss of the first 20% and the last 20% epochs
    start_average = float(start_average) / (self.epoch * 0.2)
    end_average = float(end_average) / (self.epoch * 0.2)
    print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
      % (start_average, end_average, 100 - (100*end_average/start_average)))


  def run_test(self):
    print("[INFO] Testing . . .")
    for k in range(0, 15):
      np.random.seed(k)
      test_data, test_label = test_input_setup(self, np.random.randint(0,200))

      print("Testing...")

      start_time = time.time()
      result = np.clip(self.pred.eval({self.images: test_data, self.labels: test_label, self.batch: 1}), 0, 1)
      print(" RESULT ", result.shape)
      print(" test_data ", test_data.shape)
      print(" test_label ", test_label.shape)

      passed = time.time() - start_time
      img1 = tf.convert_to_tensor(test_label, dtype=tf.float32)
      img2 = tf.convert_to_tensor(result, dtype=tf.float32)


      psnr = self.sess.run(tf.image.psnr(img1, img2, 1))
      ssim = self.sess.run(tf.image.ssim(img1, img2, 1))
      print("Took %.3f seconds, PSNR: %.6f, SSIM: %.6f" % (passed, psnr, ssim))


      image_path = os.path.join(os.getcwd(), self.output_dir)
      image_path = os.path.join(image_path, "test_image.png")
      import matplotlib.pyplot as plt

      fig = plt.figure(figsize=(11,11))

      ax1 = fig.add_subplot(131); ax1.imshow(test_data.reshape(128,128)); ax1.axis('off'); ax1.set_title('LR')
      ax2 = fig.add_subplot(132); ax2.imshow(test_label.reshape(384,384)); ax2.axis('off'); ax2.set_title('HR')
      ax3 = fig.add_subplot(133); ax3.imshow(result.reshape(384,384)); ax3.axis('off'); ax3.set_title('Predicted')

      plt.savefig('./result/' + str(k) + '_test.png')

  def save(self, step):
    model_name = self.model.name + ".model"

    if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)

    self.saver.save(self.sess,
                    os.path.join(self.model_dir, model_name),
                    global_step=step)

  def load(self):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(self.model_dir, ckpt_name))
        return True
    else:
        return False
