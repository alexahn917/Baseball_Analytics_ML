# Author: SangHyeon (Alex) Ahn

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

# Perceptron class used for multi classifications of instances
# Iterates all intstances 'It' times
class MC_Perceptron(Predictor):
    def __init__(self, It):
        self.IT = It
        self.w = {} # KxM weight vector (class_label_k, w_k)

    def train(self, instances):
        # Initialize
        labels = []
        for instance in instances:
            if instance.get_label().get_int_label() not in labels:
                labels.append(instance.get_label().get_int_label())
        for label in labels:
            self.w[label] = {}
        # Train
        for i in range(self.IT):
            for instance in instances:
                prediction = self.predict(instance)
                if (prediction.get_int_label() != instance.get_label().get_int_label()):
                    self.update(instance, prediction.get_int_label(), -1)
                    self.update(instance, instance.get_label().get_int_label(), 1)

    def predict(self, instance):
        weights = []
        for label in self.w:
            weights.append([label, self.dotProd(instance, label)])
        weights.sort(key = lambda tup:tup[1], reverse = True)
        argmax = weights.pop(0)[0]
        return ClassificationLabel(argmax)

    def dotProd(self, instance, label):
        total_sum = 0
        for feature, value in instance.get_feature_vector():
            if feature in self.w[label]:
                total_sum += value * self.w[label][feature]
        return total_sum

    def update(self, instance, label, direction):
        for feature, value in instance.get_feature_vector():
            if feature not in self.w[label]:
                self.w[label][feature] = 0
            self.w[label][feature] += direction * value