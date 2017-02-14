import csv
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import tensorflow as tf

def getData():
    with open('data/dat.csv') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

    return data

def preprocessData(data):
    label_encoder = preprocessing.LabelEncoder()
    one_hot_encoder = preprocessing.OneHotEncoder()

    data[:,0] = label_encoder.fit_transform(data[:,0])
    data = data.astype(float)

    # Uncomment lines below to use stock symbol and day parameters
    # WARNING: Epochs may be extremely slow
    # processed_data = one_hot_encoder.fit_transform(data[:,0:2]).toarray()
    # processed_data = np.append(processed_data, data[:,2:6], 1)

    # Do not use stock symbol and day parameters for training
    processed_data = data[:,2:6]

    processed_data = preprocessing.normalize(processed_data)
    np.random.shuffle(processed_data)

    return processed_data

def learn(data):
    data = preprocessData(data)
    num_params = data.shape[1] - 1

    X = data[:,0:num_params]
    Y = data[:,num_params].reshape(-1, 1)

    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size=0.30)
    training_data_size = train_X.shape[0]

    stock_data = tf.placeholder(tf.float32, [None, num_params])
    stock_price = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.random_uniform([num_params, 1], dtype=tf.float32), name = "W")
    y = tf.matmul(stock_data, W)

    learning_rate = 1e-3
    cost_function = tf.reduce_mean(tf.pow(stock_price - y, 2))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    last_cost = 0
    tolerance = 1e-6
    epochs = 1
    max_epochs = 1e6

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

        print "Test cost: ", sess.run(cost_function, feed_dict={stock_data: test_X, stock_price: test_Y})
        test_results = sess.run(y, feed_dict={stock_data: test_X, stock_price: test_Y})

    avg_perc_error = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        avg_perc_error = avg_perc_error + abs(actual_change - predicted_change)

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    print "Average percentage error: ", avg_perc_error

def main():
    data = np.array(getData())
    learn(data)

main()
