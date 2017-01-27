# Author: SangHyeon (Alex) Ahn

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

# Perceptron class used for binary classifications of instances
# Iterates all intstances 'It' times and by learning rate of 'Lr'
class Perceptron(Predictor):
    def __init__(self, Lr, It):
        self.w = {}
        self.LR = Lr
        self.IT = It

    # Learn a model by training datasets (instances)
    def train(self, instances):
        for i in range(0, self.IT):
            for instance in instances:
                y_i = instance.get_label()
                y_hat = self.predict(instance)
                if y_hat.get_int_label() != y_i.get_int_label():
                    self.update(instance, instance.get_label())

    # Given an instance, predicts the y label using w
    def predict(self, instance):
        values_sum = 0
        FV = instance.get_feature_vector()
        # Perform dot product of spase arrays
        for feature, value in FV:
            values_sum += value * self.w.get(feature, 0)
        # Determine label by dot product sum
        if values_sum >= 0:
            return ClassificationLabel(1)
        else :
            return ClassificationLabel(0)

    def update(self, instance, y_label):
        FV = instance.get_feature_vector()
        # Update w accordingly
        for feature, value in FV:
            if y_label.get_int_label() is 1:
                y_i_t = 1
            else:
                y_i_t = -1
            self.w.update({feature: self.w.get(feature,0) + (self.LR *  y_i_t * value)})

# AVG_Perceptron predicts a value using sum of all weights through each example
# Extends Preceptron class
class AVG_Perceptron(Perceptron):
    def __init__(self, Lr, It):
        super(AVG_Perceptron, self).__init__(Lr, It)
        self.w_avg = {}

    # Learn a model by training datasets (instances), and compute w_avg
    def train(self, instances):
        for i in range(0, self.IT):
            for instance in instances:
                y_hat = self.predict_train(instance)
                if y_hat.get_int_label() != (instance.get_label()).get_int_label():
                    self.update(instance, instance.get_label())
                # Store w_avg
                for feature in self.w:
                    self.w_avg.update({feature : self.w_avg.get(feature,0) + self.w.get(feature,0)})

    # Given an instance, predicts the y label using w (used only for training the data)
    def predict_train(self, instance):
        values_sum = 0
        FV = instance.get_feature_vector()
        # Perform dot product of spase arrays
        for feature, value in FV:
            values_sum += value * self.w.get(feature, 0)
        # Determine label by dot product sum
        if values_sum >= 0:
            return ClassificationLabel(1)
        else :
            return ClassificationLabel(0)

    # Predicts classification by using w_avg
    def predict(self, instance):
        values_sum = 0
        FV = instance.get_feature_vector()
        # Perform dot product of spase arrays
        for feature, value in FV:
            values_sum += value * self.w_avg.get(feature, 0)
        # Determine label by dot product sum
        if values_sum >= 0:
            return ClassificationLabel(1)
        else :
            return ClassificationLabel(0)


# Margin_Perceptron predicts a value using weights and margin.
# Using margin, we ensure that a training example is not only labeled correctly but it is labeled correctly with a margin
# Extends Preceptron class
class Margin_Perceptron(Perceptron):
    def __init__(self, Lr, It):
        super(Margin_Perceptron, self).__init__(Lr, It)
        self._margin = 1

    # Learn a model by training datasets (instances)
    def train(self, instances):
        for i in range(0, self.IT):
            for instance in instances:
                x_i = instance.get_feature_vector()
                y_i = instance.get_label().get_int_label()
                if y_i is 0:
                    y_i = -1
                # Update if example is not classified with at least a margin
                if self.get_margin(y_i, x_i) < self._margin:
                    self.update(instance, instance.get_label())

    # Calculates a margin given y_i, x_i
    def get_margin(self, y_i, x_i):
        dot_prod = 0
        # Perform dot product of spase arrays
        for feature, value in x_i:
            dot_prod += value * self.w.get(feature, 0)
        return y_i*dot_prod