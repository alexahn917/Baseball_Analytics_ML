# Author: SangHyeon (Alex) Ahn

from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass
       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._label = label        
    def __str__(self):
        return str(self._label)
    def get_int_label(self):
        return self._label

class FeatureVector:
    def __init__(self):
        self.dict = {}
    def __iter__(self):
        return self.dict.iteritems()
    def __str__(self):
        return str(self.dict)
    def add(self, index, value):
        self.dict.update({index:value})
    def get(self, index):
        return self.dict.get(index,0)
    def getKeys(self):
        return self.dict.keys()
    def getValues(self):
        return self.dict.values()

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

    def get_feature_vector(self):
        return self._feature_vector
    def get_label(self):
        return self._label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass