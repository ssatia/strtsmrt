import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

with open('data/dat.csv') as csv_file:
    reader = csv.reader(csv_file)
    data = list(reader)

data = np.array(data)
np.random.shuffle(data)

X = data[:,1:5]
Y = data[:,5]
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.30)
training_data_size = train_X.shape[0]

stock_data = tf.placeholder(tf.float32, [None, 4])
stock_price = tf.placeholder("float")

W = tf.Variable(tf.zeros([4, 1], dtype=tf.float32), name="W")
y = tf.matmul(stock_data, W)

learning_rate = 0.1
cost_function = tf.reduce_mean(tf.pow(stock_price - y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

last_cost = 0
tolerance = 10
epochs = 1
max_epochs = 100

sess = tf.Session()
with sess.as_default():
    init = tf.initialize_all_variables()
    sess.run(init)

    while True:
        sess.run(optimizer, feed_dict={stock_data: train_X, stock_price: train_Y})

        if epochs % 100 == 0:
            cost = sess.run(cost_function, feed_dict={stock_data: train_X, stock_price: train_Y})
            print "Epoch: %d: Error: %.4f" %(epochs, cost)

            if abs(cost - last_cost) <= tolerance or epochs > max_epochs:
                print "Converged."
                break
            last_cost = cost

        epochs += 1

    print "Test cost: ", sess.run(cost_function, feed_dict={stock_data: train_X, stock_price: train_Y})
