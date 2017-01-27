# Author: SangHyeon (Alex) Ahn

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

class Pegasos(Predictor):
    def __init__(self, It, Lambda):
        self._w = {}
        self._iterations = It
        self._Lambda = Lambda

    def train(self, instances):
        t = 1
        for i in range(self._iterations):
            for instance in instances:
                self.update(instance, t)
                t += 1

    def predict(self, instance):
        margin = self.dot_product(self._w, instance.get_feature_vector())
        if margin >= 0:
            return ClassificationLabel(1)
        else:
            return ClassificationLabel(0)

    def update(self, instance, t):
        if instance.get_label().get_int_label() is 1:
            update_sign = 1
        else:
            update_sign = -1

        instance_features = instance.get_feature_vector()
        coefficient = (1.0 - 1.0/t)
        learning_rate = 1.0/(self._Lambda*t)
        indication = self.indicator_function(instance_features, update_sign)

        for w_key in self._w:
            updated_w = {w_key : coefficient*self._w.get(w_key)}
            self._w.update(updated_w)

        if indication is not 0:
            for x_key, x_value in instance_features:
                updated_tuple = {x_key : self._w.get(x_key,0) + learning_rate*indication*update_sign*x_value}
                self._w.update(updated_tuple)

    def indicator_function(self, instance_features, y_i_t):
        if y_i_t*self.dot_product(self._w, instance_features) < 1:
            return 1
        else:
            return 0

    def dot_product(self, vector_1, vector_2):
        val = 0
        for key_2, val_2 in vector_2:
            val += vector_1.get(key_2,0)*val_2
        return val