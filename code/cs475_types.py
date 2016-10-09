from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    """
    A label object encodes the label for a learning example. The Label class
    is abstract and you will implement a ClassificationLabel, which contains an int
    to indicate a class (binary prediction will be 0 or 1). ClassificationLabel should
    implement str so that it returns its int value as a string.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self.label = label
        
    def __str__(self):
        return str(self.label)

class FeatureVector:
    """
    The data representing an instance are stored as a feature vector.
    A feature vector is a vector of doubles, where the value of the ith dimension of the
    feature vector corresponds to the value of the ith feature. A FeatureVector must
    support operations such as get(index), which returns the value of the feature at
    index and add(index, value), which sets the value of the feature at index.
    
    Since many learning applications encode instances as sparse vectors, FeatureVector
    should be a sparse vector. A sparse vector efficiently encodes very high dimen-
    sional data by not maintaining values for features with 0 values. Some common
    implementations of sparse vectors include hash maps and lists of index/value pairs.
    If you fail to do this correctly, your code will run very slowly. You will need to add
    a method(s) to iterate over the non-empty positions of the vector. How you chose
    to do this is up to you.

    You are also welcome to use data structures within Scipy: http://www.scipy.org/.
    """
    def __init__(self):
        self.feature_vector = {}
        
    def add(self, index, value):
        self.feature_vector[index] = value
                
    def get(self, index):
        return self.feature_vector.get(index) if self.feature_vector.get(index) != None else 0.0
        

class Instance:
    """
    An instance represents a single learning example. An instance contains a
    data object and a label. The data object is a FeatureVector and the label will be
    a Label object. For classification, the label will be a ClassificationLabel object.
    When the label is unknown (test data) the label will be None.
    """
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    """
    This is an abstract class that will be the parent class for all learning
    algorithms. Learning algorithms must implement the train and predict methods.
    Predictors must be serializable using Pickle so that they can be saved after training.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
