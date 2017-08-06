from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import generate_data
import os
import time
import tensorflow as tf

import model, config

learning_rate = config.learning_rate
max_steps = config.max_steps
batch_size = config.batch_size
num_steps = config.num_steps
vocab_size = config.vocab_size
log_dir = config.log_dir


def placeholder_inputs(batch_size, time_step, feature_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, time_step,
                                                           feature_size))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size, time_step))
    return images_placeholder, labels_placeholder


def fill_feed_dict(images_pl, labels_pl, inputs, targets, start_index, batch_size):
    images_feed, labels_feed = inputs[start_index:start_index + batch_size], targets[
                                                                             start_index:start_index + batch_size]
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            inputs, targets):
    """Runs one evaluation against the full epoch of data.

    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = len(inputs) // batch_size
    num_examples = steps_per_epoch * batch_size
    start_index = 0;
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(images_placeholder, labels_placeholder, inputs, targets, start_index, batch_size)
        start_index = (start_index + batch_size)

        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(batch_size, num_steps, vocab_size)

        logits = model.inference(images_placeholder, batch_size)
        loss = model.loss(logits, labels_placeholder, batch_size)
        train_op = model.training(loss, learning_rate)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        eval_correct = model.evaluation(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        sess.run(init)

        inputs_train, targets_train = generate_data.genernate_random_data1(config.train_size)
        inputs_test, targets_test = generate_data.genernate_random_data2(config.test_size)

        start_index = 0
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(images_placeholder, labels_placeholder, inputs_train, targets_train, start_index,
                                       batch_size)
            start_index += batch_size
            if start_index >= len(inputs_train):
                start_index = 0

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Train Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        inputs_train, targets_train)
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        inputs_test, targets_test)


def main():
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    run_training()


if __name__ == '__main__':
    main()
