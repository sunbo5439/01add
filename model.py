from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import inspect
import tensorflow as tf

import config

unit_size = config.unit_size
num_layers = config.num_layers
num_steps = config.num_steps

vocab_size = config.vocab_size


def data_type():
    return tf.float32


def inference(image_placeholder, batch_size):
    inputs = image_placeholder



    def lstm_cell():
        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                unit_size, forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                unit_size, forget_bias=0.0, state_is_tuple=True)

    attn_cell = lstm_cell
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_layers)], state_is_tuple=True)
    _initial_state = cell.zero_state(batch_size, data_type())
    outputs = []
    state = _initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            temp=inputs[:, time_step, :]
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, unit_size])
    softmax_w = tf.get_variable(
        "softmax_w", [unit_size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
    return logits


def loss(logits, labels_placeholder,batch_size):
    targets = labels_placeholder
    # use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )
    return tf.reduce_sum(loss)


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    labels_pre = tf.argmax(logits, -1)
    labels_pre = tf.cast(labels_pre,tf.int32)
    correct = tf.reduce_all(tf.equal(labels_pre, labels), -1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
