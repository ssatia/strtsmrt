import csv
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import tensorflow as tf

def getData():
    with open('data/dat_0.csv') as csv_file:
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

    # Split the data into training, validation, and testing sets (60/20/20)
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size = 0.40)
    test_X, valid_X, test_Y, valid_Y = cross_validation.train_test_split(test_X, test_Y, test_size = 0.50)

    # Get the initial stock prices for computing the relative cost
    train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
    valid_opening_price = valid_X[:, num_params - 1].reshape(-1, 1)
    test_opening_price = test_X[:, num_params - 1].reshape(-1, 1)

    stock_data = tf.placeholder(tf.float32, [None, num_params])
    opening_price = tf.placeholder(tf.float32, [None, 1])
    stock_price = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.random_uniform([num_params, 1], dtype=tf.float32), name = "W")
    y = tf.matmul(stock_data, W)

    learning_rate = 1e-3
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.subtract(stock_price, y), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_train_cost = 0
    best_valid_cost = 1e6
    best_valid_epoch = 0
    valid_epoch_threshold = 1
    tolerance = 1e-10
    epochs = 1
    max_epochs = 1e6
    # saver = tf.train.Saver([W])

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while True:
            sess.run(optimizer, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y})

            if epochs % 100 == 0:
                train_cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y})
                valid_cost = sess.run(cost_function, feed_dict={stock_data: valid_X, opening_price: valid_opening_price, stock_price: valid_Y})
                print "Epoch: %d: Training error: %f Validation error: %f" %(epochs, train_cost, valid_cost)

                if(valid_cost < best_valid_cost):
                    best_valid_cost = valid_cost
                    best_valid_epoch = epochs
                    # save_path = saver.save(sess, 'lr-model-valid')

                if(valid_epoch_threshold <= epochs - best_valid_epoch):
                    # saver.restore(sess, save_path)
                    print "Early stopping."
                    break

                if abs(train_cost - last_train_cost) <= tolerance or epochs > max_epochs:
                    print "Converged."
                    break

                last_train_cost = train_cost

            epochs += 1

        print "Test error: ", sess.run(cost_function, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y})
        test_results = sess.run(y, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y})

    avg_perc_error = 0
    max_perc_error = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        delta = abs(actual_change - predicted_change)
        max_perc_error = max(max_perc_error, delta)
        avg_perc_error = avg_perc_error + delta

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    max_perc_error *= 100
    print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)

def main():
    data = np.array(getData())
    learn(data)

main()
