import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import tensorflow as tf
import numpy as np
import pandas as pd

def readCSV():
    pitcher_name = 'Chris Sale'
    train_csv_file = '../ETL pipeline/CSV/raw/' + pitcher_name + '_train' + '.csv'
    test_csv_file = '../ETL pipeline/CSV/raw/' + pitcher_name + '_test' + '.csv'
    X_train = pd.DataFrame.from_csv(train_csv_file, index_col=None)
    y_train = X_train['pitch_type']
    X_train.drop('pitch_type', axis = 1, inplace=True)
    X_test = pd.DataFrame.from_csv(test_csv_file, index_col=None)
    y_test = X_test['pitch_type']
    X_test.drop('pitch_type', axis = 1, inplace=True)
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = readCSV()
    #data_train = [X_train, np.eye(n_labels)[y_train]]
    #data_train = [data[0][:cut], np.eye(n_labels)[data[1][:cut]]]
    #data_test = [data[0][cut:], np.eye(n_labels)[data[1][cut:]]]
    n_labels = len(np.unique(y_test))
    n_samples = len(X_train)
    n_features = len(X_train.iloc[0])
    print("Number of labels: %d" %n_labels)
    print("Number of features: %d" %n_features)
    print("Number of instances: %d" %n_samples)

    # declare variables
    x = tf.placeholder(tf.float32, [None, n_features])
    W = tf.Variable(tf.zeros([n_features, n_labels]))
    b = tf.Variable(tf.zeros([n_labels]))
    y_ = tf.placeholder(tf.float32, [None, n_labels])

    # softmax
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # cross entropy loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # backpropagation
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    index_in_epoch = 0

    LABELS = {}
    index = 0
    for label in np.unique(y_train):
        LABELS[label] = index
        index+=1

    int_y_test = []
    int_y_train = []
    for i in y_test:
        int_y_test.append(LABELS[i])
    for i in y_train:
        int_y_train.append(LABELS[i])
    y_test = int_y_test
    y_train = int_y_train
    y_test = np.eye(n_labels)[y_test]
    y_train = np.eye(n_labels)[y_train]

    X_train, y_train = shuffle(X_train, y_train)

    # train by batch iterations
    iterations = 10000
    for i in range(iterations):
        #print "EPOCH: %d" % index_in_epoch
        if index_in_epoch > n_samples:
            X_train, y_train = shuffle(X_train, y_train)
            index_in_epoch = 0
        batch_xs, batch_ys, index_in_epoch = next_batch(X_train, y_train, 100, index_in_epoch)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))

def shuffle(X_train,y_train):
    new_index = np.random.permutation(X_train.index)
    X_train = X_train.reindex(new_index)
    new_y_train = [y_train[i] for i in new_index]
    return X_train, new_y_train


def next_batch(X, y, batch_size, index_in_epoch):
    """Return the next `batch_size` examples from this data set."""
    
    # get next batch
    start = index_in_epoch
    index_in_epoch += batch_size
    end = index_in_epoch    
    return X.iloc[start:end], y[start:end], index_in_epoch

if __name__ == "__main__":
    main()
