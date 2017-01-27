# Author: SangHyeon (Alex) Ahn
import math
from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

class KNN(Predictor):
    def __init__(self, k):
        self.K = k
        self.train_instances = []
        self.max_feature = 0

    # store all train instances for distance calculations
    def train(self, instances):
        max_feature_num = 0
        self.train_instances = instances
        for instance in instances:
            max_feature_num = max(instance.get_feature_vector().getKeys(), key=int)
            if max_feature_num > self.max_feature:
                self.max_feature = max_feature_num

    # make predictions based on k-nn labels
    def predict(self, instance):
        Distances = []
        for train_instance in self.train_instances:
            Distances.append([self.distance(instance, train_instance), train_instance.get_label().get_int_label()])
        KNN_labels = []
        Distances.sort()

        for i in range(self.K):
            min_distance = Distances.pop(0)
            KNN_labels.append(min_distance[1])
        return self.dominant_label(KNN_labels)

    # get dominat label from knn
    def dominant_label(self, labels):
        counts = {}
        for label in labels:
            counts.update({label : counts.get(label,0) + 1})
        max_count = 0
        max_count_label = 0
        for key, value in counts.items():
            if max_count < value:
                max_count = value
                max_count_label = key
            if value is max_count and key < max_count_label:
                max_count_label = key
        return ClassificationLabel(max_count_label)

    # calculates a distance between two instances (two sparse feature vectors) 
    def distance(self, instance_i, instance_j):
        feature_vector_i = instance_i.get_feature_vector()
        feature_vector_j = instance_j.get_feature_vector()
        dist_vector = []
        for i in range(self.max_feature + 1):
            dist_vector.append((feature_vector_i.get(i) - feature_vector_j.get(i))**2)
        return math.sqrt(sum(dist_vector))

class Distance_KNN(KNN):
    def __init__(self, k):
        super(Distance_KNN, self).__init__(k)

    # make predictions based on k-nn labels and their distances
    def predict(self, instance):
        Distances = []

        for train_instance in self.train_instances:
            Distances.append([self.distance(instance, train_instance), train_instance.get_label().get_int_label()])
        Distances.sort()


        KNN_labels = []
        for i in range(self.K):
            min_distance = Distances.pop(0)
            KNN_labels.append([min_distance[0], min_distance[1]])
        return self.dominant_label(KNN_labels)

    # get dominat label from knn and their distances
    def dominant_label(self, KNN_labels):
        counts = {}
        for dist, int_label in KNN_labels:
            counts.update({int_label : counts.get(int_label,0) + 1/(1+dist**2)})
        max_count = 0
        max_count_label = 0
        for key, value in counts.items():
            if max_count < value:
                max_count = value
                max_count_label = key
            if value is max_count and key < max_count_label:
                max_count_label = key
                print("broke ties!")
        return ClassificationLabel(max_count_label)