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

    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size=0.30)
    train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
    test_opening_price = test_X[:, num_params - 1].reshape(-1, 1)

    stock_data = tf.placeholder(tf.float32, [None, num_params])
    opening_price = tf.placeholder(tf.float32, [None, 1])
    stock_price = tf.placeholder(tf.float32, [None, 1])

    # Number of neurons in the hidden layer
    n_hidden_1 = 3
    n_hidden_2 = 3

    weights = {
        'h1': tf.Variable(tf.random_normal([num_params, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([1]))
    }

    layer_1 = tf.add(tf.matmul(stock_data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    output_layer = tf.matmul(layer_2, weights['out'])

    learning_rate = 1e-3
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.sub(stock_price, output_layer), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_cost = 0
    tolerance = 1e-6
    epochs = 1
    max_epochs = 1e6

    sess = tf.Session()
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)

        while True:
            sess.run(optimizer, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y})

            if epochs % 100 == 0:
                cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y})
                print "Epoch: %d: Error: %f" %(epochs, cost)

                if abs(cost - last_cost) <= tolerance or epochs > max_epochs:
                    print "Converged."
                    break
                last_cost = cost

            epochs += 1

        print "Test error: ", sess.run(cost_function, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y})
        test_results = sess.run(output_layer, feed_dict={stock_data: test_X, stock_price: test_Y})

    avg_perc_error = 0
    max_perc_error = 0
    mei = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        delta = abs(actual_change - predicted_change)
        avg_perc_error = avg_perc_error + delta
        if delta > max_perc_error:
            max_perc_error = delta
            mei = i

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    max_perc_error *= 100
    print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)\

def main():
    data = np.array(getData())
    learn(data)

main()
