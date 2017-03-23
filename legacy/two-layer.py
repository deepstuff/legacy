import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split

# load dataset
dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename='../data/legacy-vectors.csv',
                                                                 target_dtype=np.float32,
                                                                 features_dtype=np.float32)

# turn target into one-hot
target = np.array([[1 if y == 0 else 0, 1 if y == 1 else 0] for y in dataset.target])

# split in to train and test sets
trX, teX, trY, teY = train_test_split(dataset.data, target, test_size=0.33, stratify=dataset.target, random_state=1)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

X = tf.placeholder("float", [None, 585])
Y = tf.placeholder("float", [None, 2])

w_h = init_weights([585, 10]) # two-layer network, 10 hidden nodes
w_o = init_weights([10, 2])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(10000):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})))
