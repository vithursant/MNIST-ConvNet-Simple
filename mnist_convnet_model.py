import tensorflow as tf
import os
import time
from tqdm import trange, tqdm

from tensorflow.examples.tutorials.mnist import input_data

# Define current time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

flags = tf.app.flags

# Define Directory Parameters
flags.DEFINE_string('data_dir', os.getcwd() + '/data/', 'Directory for data')
flags.DEFINE_string('log_dir', os.getcwd() + '/log/', 'Directory for logs')
flags.DEFINE_string('checkpoint_dir',
                    os.getcwd() + '/checkpoint/' + current_time,
                    'Directory for checkpoints')

# Define Model Parameters
flags.DEFINE_integer('batch_size', 128, 'Minibatch size')
flags.DEFINE_integer('num_iters', 200000, 'Number of iterations')
flags.DEFINE_float('learning_rate', 3e-3, 'Learning rate')
flags.DEFINE_integer('num_classes', 10, 'Number of classes (0-9 digits)')
flags.DEFINE_integer('num_input', 784, 'MNIST data input (img shape: 28*28)')
flags.DEFINE_float('dropout', 0.75, 'Dropout, probability to keep units')
flags.DEFINE_integer('display_step', 100, 'Display step')

FLAGS = flags.FLAGS
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# tf Graph input
def init_graph_inputs():
    # input, i.e. pixels that constitute the image
    x = tf.placeholder(tf.float32, [None,FLAGS.num_input])

    # labels, i.e. which digit the image is
    y = tf.placeholder(tf.float32, [None,FLAGS.num_classes])

    # keep probability (dropout)
    dropout_prob = tf.placeholder(tf.float32)

    return x, y, dropout_prob

# Define utility functions weight and biases
def weight_variable():
    weights = {
                  'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                  'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                  'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
                  'out': tf.Variable(tf.random_normal([1024, FLAGS.num_classes]))
              }

    return weights

def bias_variable():
    biases = {
                 'bc1': tf.Variable(tf.random_normal([32])),
                 'bc2': tf.Variable(tf.random_normal([64])),
                 'bd1': tf.Variable(tf.random_normal([1024])),
                 'out': tf.Variable(tf.random_normal([FLAGS.num_classes]))
             }

    return biases

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x,
                          ksize=[1, k, k, 1],
                          strides=[1, k, k, 1],
                          padding="SAME")

# Create ConvNet model
def conv_net(x, weights, biases, dropout):
    # Reshape input structure
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution layer #1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling) #1
    conv1 = maxpool2d(conv1, k=2)

    # Convolution layer #2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling) #2
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

def loss_optimizer(y, pred):
    #print(y.get_shape())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

    return cost, optimizer

def evaluate_model(y, pred):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return correct_pred, accuracy

def construct_model(x, weights, biases, keep_prob):
    pred = conv_net(x, weights, biases, keep_prob)

    return pred

def train():
    x, y, keep_prob = init_graph_inputs()
    weights = weight_variable()
    biases = bias_variable()

    pred = construct_model(x, weights, biases, keep_prob)

    #y = tf.Print(y, [y], message="This is a: ")
    #pred.eval()
    #exit()

    cost, optimizer = loss_optimizer(y, pred)

    correct_pred, accuracy = evaluate_model(y, pred)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Keep training until reach max iterations
        for i in tqdm(xrange(1, FLAGS.num_iters)):
            batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x,
                                           y: batch_y,
                                           keep_prob: FLAGS.dropout})

            if i % FLAGS.display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(i * FLAGS.batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            if i * FLAGS.batch_size >= FLAGS.num_iters:
                break

        print("Optimization finished")

        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                           y: mnist.test.labels[:256],
                                           keep_prob: 1.}))

def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.data_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    train()

if __name__=="__main__":
    main()
