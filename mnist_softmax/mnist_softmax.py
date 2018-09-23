#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main ( _ ) :
    # import data
    mnist = input_data.read_data_sets ( FLAGS.data_dir )

    # create the model
    x = tf.placeholder ( tf.float32, [ None, 784 ] )
    W = tf.Variable ( tf.zeros ( [ 784, 10 ] ) )
    b = tf.Variable ( tf.zeros ( [ 10 ] ) )
    y = tf.matmul ( x, W ) + b

    # define loss and optimizer

    y_ = tf.placeholder ( tf.int64, [ None ] )

    cross_entropy = tf.losses.sparse_softmax_cross_entropy ( labels = y_, logits = y )
    train_step = tf.train.GradientDescentOptimizer ( 0.5 ).minimize ( cross_entropy )

    sess = tf.InteractiveSession ()
    tf.global_variables_initializer ().run ()

    
