import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorboardcolab import *
import argparse

LOGDIR = "/tmp/mnist_tutorial/"
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def weight_variable(shape):
  '''Create a weight variable with initialization'''
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="W")

def bias_variable(shape):
  '''Create a bias variable with initialization'''
  return tf.Variable(tf.constant(0.1, shape=shape), name="B")

def conv2d(x, weights):
  '''Perform 2d convolution with specified weights'''
  return tf.nn.conv2d(input=x, filter=weights, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
  '''Perform max pooling'''
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, kernel_size, in_channels, out_channels, batch_norm=False, name='conv'):
  '''Convolutional layer with 2d convolution, relu activation, and max pooling'''
  with tf.name_scope(name):
    w = weight_variable([kernel_size, kernel_size, in_channels, out_channels])
    b = bias_variable([out_channels])
    if not batch_norm:
        activation = tf.nn.relu(conv2d(input, w) + b)
    else:
        activation = tf.nn.relu(tf.nn.batch_normalization(conv2d(input, w)+b, 0,1,0.1,10,0.0001))
    tf.summary.histogram("Weights", w)
    tf.summary.histogram("Biases", b)
    tf.summary.histogram("Activations", activation)
    pool = max_pool(activation)
    return pool

def fc_layer(input, in_channels, out_channels, name='fc'):
  '''Fully connected layer'''
  with tf.name_scope(name):
    w = weight_variable([in_channels, out_channels])
    b = bias_variable([out_channels])
    out = tf.matmul(input, w) + b
    tf.summary.histogram("Weights", w)
    tf.summary.histogram("Biases", b)
    tf.summary.histogram("Activations", out)
    return out

def train():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test', default=False, type=bool)
  parser.add_argument('--batch_norm', default=False, type=bool)
  args = parser.parse_args()

  tf.reset_default_graph()
  sess = tf.Session()
  # Input flattened image
  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  # Class label
  y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

  x_image = tf.reshape(x, [-1,28,28,1])

  # Modified Model
  conv_out1 = conv_layer(x_image, 5, 1, 8, args.batch_norm, "conv1")
  conv_out2 = conv_layer(conv_out1, 5, 8, 64, args.batch_norm, "conv2")
  flattened_out = tf.reshape(conv_out2, [-1, 7*7*64])
  fc_out1 = fc_layer(flattened_out, 7*7*64, 1024, "fc1")
  relu = tf.nn.relu(fc_out1)
  dropout = tf.layers.dropout(inputs=relu, rate=0.4)
  logits = fc_layer(dropout, 1024, 10, "fc2")

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
  tf.summary.scalar("Loss", loss)

  optimizer = tf.train.AdamOptimizer(0.001)
  train_step = optimizer.minimize(loss)

  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  train_acc = tf.summary.scalar('Accuracy', accuracy)
  validation_acc = tf.summary.scalar('Validation_accuracy', accuracy)

  merged_summary = tf.summary.merge_all()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter('/tmp/mnist_tutorial')
  writer.add_graph(sess.graph)

  for epoch in range(2001):
    batch = mnist.train.next_batch(50)
    if epoch % 100 == 0:
    #   train_summary_str = sess.run(train_acc, feed_dict={x: batch[0], y: batch[1]})
      [summary, train_accuracy] = sess.run([train_acc, accuracy], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(summary, epoch)
      print('Epoch %d, train accuracy: %f'%(epoch, train_accuracy))
      [validation_summary, val_accuracy] = sess.run([validation_acc, accuracy], \
                        feed_dict={x: mnist.validation.images, y:mnist.validation.labels})
      writer.add_summary(validation_summary, epoch)
      print('Epoch %d, validation accuracy: %f'%(epoch, val_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
    if args.test:
        if epoch == 801:
            break
  if args.test:
    print('Test accuracy: %g'%accuracy.eval(session=sess, \
                            feed_dict={x: mnist.test.images, y:mnist.test.labels}))
  writer.close()
  print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

train()
