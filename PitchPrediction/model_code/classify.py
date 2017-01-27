# Author: SangHyeon (Alex) Ahn

import os
import argparse
import sys
import pickle

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
from Perceptron import Perceptron, AVG_Perceptron, Margin_Perceptron
from Pegasos import Pegasos
from KNN import KNN, Distance_KNN
from AdaBoost import AdaBoost
from LambdaMeans import LambdaMeans
from NB_Cluster import NB_Cluster
from MC_Perceptron import MC_Perceptron

def load_data(filename):
    instances = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
            
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")
    parser.add_argument("--online-learning-rate", type=float, help="The learning rate for perceptron", default=1.0)
    parser.add_argument("--online-training-iterations", type=int, help="The number of training iterations for online methods", default=5)
    parser.add_argument("--pegasos-lambda", type=float, help="The regularization parameter for Pegasos.", default=1e-4)
    parser.add_argument("--knn", type=int, help="The value of K for KNN classification.", default=10)
    parser.add_argument("--num-boosting-iterations", type=int, help="The number of boosting iterations to run.", default=10)
    parser.add_argument("--cluster-lambda", type=float, help="The value of lambda in lambda-means.", default=0.0)
    parser.add_argument("--num-clusters", type=int, help="The number of clusters in Naive Bayes clustering.", default=3)
    parser.add_argument("--clustering-training-iterations", type=int, help="The number of clustering iterations.", default=10)
    args = parser.parse_args()
    check_args(args)
    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(instances, algorithm):
    args = get_args()
    LR = args.online_learning_rate
    IT = args.online_training_iterations
    Pegasos_Lambda = args.pegasos_lambda
    K = args.knn
    T = args.num_boosting_iterations
    cluster_lambda = args.cluster_lambda
    cluster_iterations = args.clustering_training_iterations
    num_clusters = args.num_clusters

    if algorithm == 'perceptron':
        perceptron = Perceptron(LR, IT)
        perceptron.train(instances)
        return perceptron

    elif algorithm == 'averaged_perceptron':
        avg_perceptron = AVG_Perceptron(LR, IT)
        avg_perceptron.train(instances)
        return avg_perceptron

    elif algorithm == 'margin_perceptron':
        margin_perceptron = Margin_Perceptron(LR, IT)
        margin_perceptron.train(instances)
        return margin_perceptron

    elif algorithm == 'pegasos':
        pegasos = Pegasos(IT, Pegasos_Lambda)
        pegasos.train(instances)
        return pegasos

    elif algorithm == 'knn':
        Knn = KNN(K);
        Knn.train(instances)
        return Knn

    elif algorithm == 'distance_knn':
        Dist_Knn = Distance_KNN(K)
        Dist_Knn.train(instances)
        return Dist_Knn

    elif algorithm == 'adaboost':
        Adaboost_predictor = AdaBoost(T)
        Adaboost_predictor.train(instances)
        return Adaboost_predictor

    elif algorithm == 'lambda_means':
        LambdaMeans_predictor = LambdaMeans(cluster_lambda, cluster_iterations)
        LambdaMeans_predictor.train(instances)
        return LambdaMeans_predictor

    elif algorithm == 'nb_clustering':
        NB_Cluster_predictor = NB_Cluster(num_clusters, cluster_iterations)
        NB_Cluster_predictor.train(instances)
        return NB_Cluster_predictor

    elif algorithm == 'mc_perceptron':
        MC_Perceptron_predictor = MC_Perceptron(IT)
        MC_Perceptron_predictor.train(instances)
        return MC_Perceptron_predictor

    else:
        print("Invalid Algorithm")
    
    return None


def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
        
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        instances = load_data(args.data)

        # Train the model.
        predictor = train(instances, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        instances = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

