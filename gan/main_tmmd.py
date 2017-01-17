import os
import numpy as np

from model_tmmd import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("max_iteration", 400000, "Epoch to train [400000]")
flags.DEFINE_float("learning_rate", 2, "Learning rate [2]")
flags.DEFINE_float("decay_rate", 1.0, "Decay rate for learning rate [1.0]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("init", 0.02, "Initialization value[0.02]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1000, "The size of batch images [1000]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("name", "tmmd_test", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_tmmd", "Directory name to save the checkpoints [checkpoint_tmmd]")
flags.DEFINE_string("sample_dir", "samples_tmmd", "Directory name to save the image samples [samples_tmmd]")
flags.DEFINE_string("log_dir", "logs_tmmd", "Directory name to save the tensorboard log [logs_tmmd]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("use_kernel", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    sample_dir_ = os.path.join(FLAGS.sample_dir, FLAGS.name)
    checkpoint_dir_ = os.path.join(FLAGS.checkpoint_dir, FLAGS.name)
    log_dir_ = os.path.join(FLAGS.log_dir, FLAGS.name)
    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    with tf.Session() as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=28, c_dim=1,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=checkpoint_dir_, sample_dir=sample_dir_, log_dir=log_dir_)
        else:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.sampling(FLAGS)

        if FLAGS.visualize:
            to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                          [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                          [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                          [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                          [dcgan.h4_w, dcgan.h4_b, None])

            # Below is codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
