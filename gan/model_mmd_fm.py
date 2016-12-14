from __future__ import division, print_function
from glob import glob
import os
import time

import numpy as np
import scipy.misc
from six.moves import xrange
import tensorflow as tf

from mmd import mix_rbf_mmd2
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import save_images


class DCGAN(object):
    def __init__(self, sess, config, is_crop=True,
                 batch_size=64, output_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, log_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.config = config
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.sample_size = batch_size
        self.output_size = output_size
        self.sample_dir = sample_dir
        self.log_dir=log_dir
        self.checkpoint_dir = checkpoint_dir
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.build_model()


    def imageRearrange(self, image, block=4):
        image = tf.slice(image, [0, 0, 0, 0], [block * block, -1, -1, -1])
        x1 = tf.batch_to_space(image, [[0, 0], [0, 0]], block)
        image_r = tf.reshape(tf.transpose(tf.reshape(x1,
            [self.output_size, block, self.output_size, block, self.c_dim])
            , [1, 0, 3, 2, 4]),
            [1, self.output_size * block, self.output_size * block, self.c_dim])
        return image_r


    def build_model(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        tf.summary.histogram("z", self.z)

        self.G = self.generator_mnist(self.z)
        images = tf.reshape(self.images, [self.batch_size, -1])
        G = tf.reshape(self.G, [self.batch_size, -1])

        phi_images = self.discriminator_k(images)
        bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

        if self.config.use_weighted_layer_kernel:
            n_layer = len(phi_images)
            for layer_id in range(n_layer):
                with tf.variable_scope("dk_att_" + str(layer_id)):
                    dk_att = tf.get_variable(
                        "weight", [len(bandwidths)], tf.float32,
                        tf.constant_initializer(1 / len(bandwidths)))
        phi_G = self.discriminator_k(G, reuse=True)
        self.kernel_loss = tf.Variable(0.0, trainable=False, name="kernel_loss")

        if self.config.use_weighted_layer_kernel:
            n_layer = len(phi_images)
            if self.config.use_gan:
                n_layer -= 1
            tf.summary.histogram("phi_images", tf.concat(1, phi_images[1:]))
            tf.summary.histogram("phi_G", tf.concat(1, phi_G[1:]))

            for layer_id in range(n_layer):
                self.kernel_loss += mix_rbf_mmd2(
                    phi_G[layer_id], phi_images[layer_id],
                    sigmas=bandwidths,
                    wts=tf.exp(dk_att) / tf.reduce_sum(tf.exp(dk_att)))
        elif self.config.use_layer_kernel:
            n_layer = len(phi_images)
            for layer_id in range(n_layer):
                kernel_loss = mix_rbf_mmd2(
                    phi_G[layer_id], phi_images[layer_id], sigmas=bandwidths)
                tf.summary.scalar("kernel_loss_" + str(layer_id), kernel_loss)
                self.kernel_loss += kernel_loss #pow(2, n_layer) * kernel_loss
            if self.config.use_gan:
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(phi_images[-1], tf.ones_like(phi_images[-1])))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(phi_G[-1], tf.zeros_like(phi_G[-1])))
                self.d_loss = self.d_loss_real + self.d_loss_fake
                tf.summary.scalar("d_loss", self.d_loss)
        else:
            if self.config.use_gan:
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(phi_images[-1], tf.ones_like(phi_images[-1])))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(phi_G[-1], tf.zeros_like(phi_images[-1])))
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(phi_G[-1], tf.ones_like(phi_images[-1])))
                self.d_loss = self.d_loss_real + self.d_loss_fake
                tf.summary.scalar("d_loss", self.d_loss)
                self.phiG = phi_G[-1]
                phi_G = tf.concat(1, phi_G[0:-1])
                phi_images = tf.concat(1, phi_images[0:-1])

                print("use_gan")
            phi_G = [phi_G]
            phi_images = [phi_images]
            n_layer = 1
            self.kernel_loss = mix_rbf_mmd2(phi_G[0], phi_images[0], sigmas=bandwidths)
        tf.summary.scalar("kernel_loss", self.kernel_loss)
        self.kernel_loss = tf.sqrt(self.kernel_loss)

        tf.summary.image("train/input image", self.imageRearrange(tf.clip_by_value(self.images, 0, 1), 8))
        tf.summary.image("train/gen image", self.imageRearrange(tf.clip_by_value(self.G, 0, 1), 8))

        self.sampler = self.generator_mnist(self.z, is_train=False, reuse=True)
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.dk_vars = [var for var in t_vars if 'dk_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            data_X, data_y = self.load_mnist()
        else:
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        if self.config.use_kernel:
            kernel_g_optim = tf.train.MomentumOptimizer(self.lr, 0.9) \
                      .minimize(self.kernel_loss, var_list=self.g_vars, global_step=self.global_step)
            if self.config.use_gan:
                kernel_d_optim = tf.train.MomentumOptimizer(self.config.kernel_d_learning_rate * self.lr, 0.9) \
                      .minimize(self.d_loss, var_list=self.dk_vars)
            else:
                kernel_d_optim = tf.train.MomentumOptimizer(self.config.kernel_d_learning_rate * self.lr, 0.9) \
                        .minimize((-1) * self.kernel_loss, var_list=self.dk_vars)

        self.sess.run(tf.global_variables_initializer())
        TrainSummary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

        if config.dataset == 'mnist':
            sample_images = data_X[0:self.sample_size]
        else:
           return
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.dataset == 'mnist':
            batch_idxs = len(data_X) // config.batch_size
        else:
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))
            batch_idxs = min(len(data), config.train_size) // config.batch_size
        lr = self.config.learning_rate
        d_loss = 0
        for it in xrange(self.config.max_iteration):
            if np.mod(it, batch_idxs) == 0:
                perm = np.random.permutation(len(data_X))
            if np.mod(it, 10000) == 1:
                lr = lr * self.config.decay_rate
            idx = np.mod(it, batch_idxs)
            batch_images = data_X[perm[idx*config.batch_size:
                                       (idx+1)*config.batch_size]]

            batch_z = np.random.uniform(
                -1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

            if self.config.use_kernel:
                _, summary_str, step, kernel_loss = self.sess.run(
                    [kernel_d_optim, TrainSummary, self.global_step,
                     self.kernel_loss],
                    feed_dict={self.lr: lr,
                               self.images: batch_images,
                               self.z: batch_z})
                _, summary_str, step, kernel_loss = self.sess.run(
                    [kernel_g_optim, TrainSummary, self.global_step,
                     self.kernel_loss],
                    feed_dict={self.lr: lr,
                               self.images: batch_images,
                               self.z: batch_z})
            counter += 1
            if np.mod(counter, 10) == 1:
                if self.config.use_gan:
                    d_loss = self.sess.run(
                        self.d_loss,
                        feed_dict={self.lr: lr,
                                   self.images: batch_images,
                                   self.z: batch_z})
                self.writer.add_summary(summary_str, step)
                print(("optmmd Epoch: [%2d] time: %4.4f, kernel_loss: %.8f, "
                       "d_loss: %.8f") %
                      (it, time.time() - start_time, kernel_loss, d_loss))
            if np.mod(counter, 500) == 1:
                self.save(self.checkpoint_dir, counter)
                samples = self.sess.run(
                    self.sampler,
                    feed_dict={self.z: sample_z, self.images: sample_images})
                print(samples.shape)
                p = os.path.join(self.sample_dir, 'train_{:02d}.png'.format(it))
                save_images(samples[:64, :, :, :], [8, 8], p)


    def sampling(self, config):
        self.sess.run(tf.global_variables_initializer())
        print(self.checkpoint_dir)
        if self.load(self.checkpoint_dir):
            print("sucess")
        else:
            print("fail")
            return
        n = 1000
        batches = n // self.batch_size
        sample_dir = os.path.join("tmp", config.name)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        for batch_id in range(batches):
            samples_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            [G] = self.sess.run([self.G], feed_dict={self.z: samples_z})
            print("G shape", G.shape)
            for i in range(self.batch_size):
                G_tmp = np.zeros((28, 28, 3))
                G_tmp[:,:,:1] = G[i]
                G_tmp[:,:,1:2] = G[i]
                G_tmp[:,:,2:3] = G[i]

                n = i + batch_id * self.batch_size
                p = os.path.join(sample_dir, "img_{}.png".format(n))
                scipy.misc.imsave(p, G_tmp)


    def discriminator_k(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        #1024, 512, 128
        h0 = tf.nn.sigmoid(linear(image, 512, 'dk_h0_lin', stddev=self.config.init))
        h1 = tf.nn.sigmoid(linear(h0, 256, 'dk_h1_lin', stddev=self.config.init))
        h2 = tf.nn.sigmoid(linear(h1, 256, 'dk_h2_lin', stddev=self.config.init))
        h3 = tf.nn.sigmoid(linear(h2, 128, 'dk_h3_lin', stddev=self.config.init))
        h4 = tf.nn.relu(linear(h3, 64, 'dk_h4_lin', stddev=self.config.init))
        if self.config.use_gan:
          h5 = linear(h4, 1, 'dk_h5_lin', stddev=self.config.init)
          return image, h0, h1, h2, h3, h4, h5
        elif self.config.use_layer_kernel:
          return image, h0, h1, h2, h3, h4
        elif self.config.use_scale_kernel:
          return tf.concat(1, [image, (28.0 * 28.0/512.0) * h0, (28 * 28.0/256.0) * h1, (28.0 * 28.0/256.0) * h2, (28.0 * 28.0/128.0) * h3,
 (28.0 * 28.0/64.0) * h4])

        else:
          return tf.concat(1, [image, h0, h1, h2, h3, h4])


    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s = self.output_size
        if np.mod(s, 16) == 0:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
        else:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = linear(tf.reshape(h1, [self.batch_size, -1]), 1, 'd_h2_lin')
            if not self.config.use_kernel:
              return tf.nn.sigmoid(h2), h2
            else:
              return tf.nn.sigmoid(h2), h2, h1, h0


    def generator_mnist(self, z, is_train=True, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = linear(z, 64, 'g_h0_lin', stddev=self.config.init)
        h1 = linear(tf.nn.relu(h0), 256, 'g_h1_lin', stddev=self.config.init)
        h2 = linear(tf.nn.relu(h1), 256, 'g_h2_lin', stddev=self.config.init)
        h3 = linear(tf.nn.relu(h2), 1024, 'g_h3_lin', stddev=self.config.init)
        h4 = linear(tf.nn.relu(h3), 28 * 28 * 1, 'g_h4_lin', stddev=self.config.init)

        return tf.reshape(tf.nn.sigmoid(h4), [self.batch_size, 28, 28, 1])


    def generator(self, z, y=None, is_train=True, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s = self.output_size
        if np.mod(s, 16) == 0:
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=is_train))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=is_train))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2, train=is_train))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3, train=is_train))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)
            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4)
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*2*s4*s4, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s4, s4, self.gf_dim * 2])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=is_train))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                [self.batch_size, s2, s2, self.gf_dim*1], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=is_train))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, s, s, self.c_dim], name='g_h2', with_w=True)

            return tf.nn.tanh(h2)


    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        return X/255.,y


    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
