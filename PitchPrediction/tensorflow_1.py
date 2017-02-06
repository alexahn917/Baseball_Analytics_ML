import csv
import tensorflow as tf
import numpy as np

def readCSV():
    csv_file = 'ETL pipeline/raw_data/Clayton Kershaw_R.csv'
    file = open(csv_file, "r")
    reader = csv.reader(file)
    instances = []
    target = []
    row_num = 0 
    for row in reader:
        if row_num is 0:
            header = row
        else:
            col_num = 0
            features = []
            for col in row:
                if col_num is 0:
                    target.append(int(col))
                    instances.append([])
                else:
                    instances[row_num-1].append(int(col))
                col_num += 1
        row_num +=1
    file.close()

    data = [instances, target]
    return data


def main():
    data = readCSV()
    n_labels = 14

    cut = int(len(data[0]) * (1/2))
    data_train = [data[0][:cut], np.eye(n_labels)[data[1][:cut]]]
    data_test = [data[0][cut:], np.eye(n_labels)[data[1][cut:]]]

    features = data_train[0]
    labels = data_train[1]
    n_samples = len(features)
    n_features = len(features[0])

    # assign variables
    x = tf.placeholder(tf.float32, [None, n_features])
    W = tf.Variable(tf.zeros([n_features, n_labels]))
    b = tf.Variable(tf.zeros([n_labels]))
    y_ = tf.placeholder(tf.float32, [None, n_labels])

    # softmax
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # cross entropy loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # backpropagation
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    index_in_epoch = 0

    data_train = shuffle(data_train)

    # train by batch iterations
    iterations = 1000
    for i in range(iterations):
        if index_in_epoch > n_samples:
            data_train = shuffle(data_train)
            index_in_epoch = 0
        batch_xs, batch_ys, index_in_epoch = next_batch(data_train, 100, index_in_epoch)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: data_test[0], y_: data_test[1]}))

def shuffle(data):
    size = len(data[0])
    new_index = np.arange(size)
    np.random.shuffle(new_index)
    features = []
    targets = []
    for i in new_index:
        features.append(data[0][i])
        targets.append(data[1][i])
    return [features, targets]

def next_batch(data, batch_size, index_in_epoch):
    """Return the next `batch_size` examples from this data set."""
    features = data[0]
    labels = data[1]
    n_samples = len(features)

    # get next batch
    start = index_in_epoch
    index_in_epoch += batch_size
    end = index_in_epoch
    return features[start:end], labels[start:end], index_in_epoch

if __name__ == "__main__":
    main()

