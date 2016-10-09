from cs475_types import Predictor, ClassificationLabel
from math import sqrt, log, exp

class Perceptron(Predictor):

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = {}

    def initialize_w(self, instances):
        for instance in instances:
            for key in instance._feature_vector.feature_vector.keys():
                if not self.w.has_key(key):
                    self.w[key] = 0

    def train(self, instances):
        self.initialize_w(instances)
        for k in range(self.iterations):
            for instance in instances:
                x_i = instance._feature_vector.feature_vector
                y_i = 1 if instance._label.label == 1 else -1
                y_hat = 1 if self.predict(instance).label == 1 else -1
                if y_i != y_hat:
                    self.update_weight(y_i, x_i)

    def update_weight(self, y_i, x_i):
        for key, value in x_i.items():
            self.w[key] += (self.learning_rate * y_i * value)

    def predict(self, instance):
        return ClassificationLabel(self.sign(instance._feature_vector.feature_vector))

    def sign(self, x_i):
        return 1 if self.compute_dot_product(x_i) >= 0 else 0

    def compute_dot_product(self, x_i):
        dot_product = 0
        for key, value in x_i.items():
            if self.w.has_key(key):
                dot_product += (self.w[key] * value)
        return dot_product


class AveragedPerceptron(Predictor):

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = {}
        self.w_avg = {}

    def train(self, instances):
        self.initialize_w(instances)
        for k in range(self.iterations):
            for instance in instances:
                x_i = instance._feature_vector.feature_vector
                y_i = 1 if instance._label.label == 1 else -1
                y_hat = 1 if self.predict(instance).label == 1 else -1
                if y_hat != y_i:
                    self.update_weight(y_i, x_i)
                for key in self.w.keys():
                    self.w_avg[key] += self.w[key]
        self.w = self.w_avg

    def initialize_w(self, instances):
        for instance in instances:
            for key in instance._feature_vector.feature_vector.keys():
                if not self.w.has_key(key):
                    self.w[key] = 0
                    self.w_avg[key] = 0
    
    def update_weight(self, y_i, x_i):
        for key, value in x_i.items():
            self.w[key] += (self.learning_rate * y_i * value)

    def predict(self, instance):
        return ClassificationLabel(self.sign(instance._feature_vector.feature_vector))

    def compute_dot_product(self, x_i):
        dot_product = 0
        for key, value in x_i.items():
            if self.w.has_key(key):
                dot_product += (self.w[key] * value)
        return dot_product

    def sign(self, x_i):
        return 1 if self.compute_dot_product(x_i) >= 0 else 0


class MarginPerceptron(Predictor):

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = {}

    def initialize_w(self, instances):
        for instance in instances:
            for key in instance._feature_vector.feature_vector.keys():
                if not self.w.has_key(key):
                    self.w[key] = 0

    def train(self, instances):
        self.initialize_w(instances)
        for k in range(self.iterations):
            for instance in instances:
                x_i = instance._feature_vector.feature_vector
                y_i = 1 if instance._label.label == 1 else -1
                if y_i * self.compute_dot_product(x_i) < 1:
                    self.update_weight(y_i, x_i)

    def update_weight(self, y_i, x_i):
        for key, value in x_i.items():
            self.w[key] += (self.learning_rate * y_i * value)

    def predict(self, instance):
        return ClassificationLabel(self.sign(instance._feature_vector.feature_vector))

    def sign(self, x_i):
        return 1 if self.compute_dot_product(x_i) >= 0 else 0

    def compute_dot_product(self, x_i):
        dot_product = 0
        for key, value in x_i.items():
            if self.w.has_key(key):
                dot_product += (self.w[key] * value)
        return dot_product


class Pegasos(Predictor):

    def __init__(self, iterations, pegasos_lambda):
        self.iterations = iterations
        self.pegasos_lambda = pegasos_lambda
        self.w = {}

    def initialize_w(self, instances):
        for instance in instances:
            for key in instance._feature_vector.feature_vector.keys():
                if not self.w.has_key(key):
                    self.w[key] = 0.0

    def train(self, instances):
        self.initialize_w(instances)
        t = 1.0
        for k in range(self.iterations):
            for instance in instances:
                y_i = 1.0 if instance._label.label == 1 else -1.0
                self.update_weight(y_i=y_i,\
                                   x_i=instance._feature_vector.feature_vector,\
                                   step_size=1.0 / (self.pegasos_lambda * t),\
                                   learning_rate=(1 - (1 / t)),\
                                   indicator=1.0 if y_i * self.compute_dot_product(instance) < 1 else 0.0)
                t += 1

    def update_weight(self, y_i, x_i, step_size, learning_rate, indicator):
        for (key, value) in self.w.items():
            self.w[key] *= learning_rate
        for (key, x_i_t) in x_i.items():
            self.w[key] += (step_size * indicator * y_i * x_i_t)

    def predict(self, instance):
        return ClassificationLabel(1) if self.compute_dot_product(instance) >= 0 else ClassificationLabel(0)

    def compute_dot_product(self, x_i):
        dot_product = 0.0
        for (key, value) in x_i._feature_vector.feature_vector.items():
            if self.w.has_key(key):
                dot_product += (self.w[key] * value)
        return dot_product


class KNN(Predictor):

    def __init__(self, knn, is_weighted):
        self.instances = []
        self.knn = knn
        self.all_features = []
        self.is_weighted = is_weighted

    def train(self, instances):
        self.instances = instances
        for instance in self.instances:
            for feature in instance._feature_vector.feature_vector.keys():
                if feature not in self.all_features:
                    self.all_features.append(feature)

    def predict(self, instance):
        nearest_neighbors = []
        for known_instance in self.instances:
            dist = self.compute_distance(known_instance, instance)
            nearest_neighbors.append((known_instance._label.label, dist))
        nearest_neighbors = sorted(nearest_neighbors, key=lambda tup: (tup[1]))[0:self.knn]
        votes = {}
        for (label, distance) in nearest_neighbors:
            if self.is_weighted:
                if votes.has_key(label):
                    votes[label] -= 1.0/(1+distance**2)
                else:
                    votes[label] = -1.0/(1+distance**2)
            else:
                if votes.has_key(label):
                    votes[label] -= 1
                else:
                    votes[label] = -1
        votes = sorted(votes.items(), key=lambda tup: (tup[1], tup[0])) # Sort by num_votes then label
        return votes[0][0]

    def compute_distance(self, known_instance, sample_instance):
        dist = 0
        for feature in self.all_features:
            known_instance_value = known_instance._feature_vector.feature_vector[feature] if known_instance._feature_vector.feature_vector.has_key(feature) else 0
            sample_instance_value = sample_instance._feature_vector.feature_vector[feature] if sample_instance._feature_vector.feature_vector.has_key(feature) else 0
            dist += (known_instance_value - sample_instance_value) ** 2
        return sqrt(dist)


class AdaBoost(Predictor):

    def __init__(self, num_boosting_iterations):
        self.num_boosting_iterations = num_boosting_iterations
        self.D = []
        self.all_labels = []
        self.all_features = []
        self.h_t_list = []
        self.a_list = []
        self.hypothesis_list = []
        self.instances = []
        self.h_cache = {}
        self.z_cache = {}

    def train(self, instances):
        self.initialize(instances)
        for t in range(self.num_boosting_iterations):
            j, c = self.get_h_t(instances)
            epsilon = self.compute_epsilon(j, c, instances)
            a_t = 0.5 * log((1 - epsilon) / epsilon)
            if a_t < 0.000001:
                break
            self.h_t_list.append((j, c))
            self.a_list.append(a_t)
            for i in range(len(instances)):
                x_i = instances[i]._feature_vector.feature_vector
                y_i = 1 if instances[i]._label.label == 1 else -1
                h_value = self.h_cache[(j, c, instances[i])] if self.h_cache.has_key((j, c, instances[i])) else self.compute_h(j, c, instances[i], instances)
                self.D[i] *= ((1.0 / self.compute_z(a_t, instances, j, c)) * exp(-a_t * y_i * h_value))

    def get_h_t(self, instances):
        min_epsilon = float('inf')
        min_j, min_c = -1, -1
        for j, c in self.hypothesis_list:
            curr_epsilon = self.compute_epsilon(j, c, instances)
            if curr_epsilon < min_epsilon:
                min_epsilon = curr_epsilon
                min_j, min_c = j, c
        return min_j, min_c

    def compute_epsilon(self, j, c, instances):
        epsilon = 0.0
        for i in range(len(instances)):
            y_i = 1 if instances[i]._label.label == 1 else -1
            h_value = self.h_cache[(j, c, instances[i])] if self.h_cache.has_key((j, c, instances[i])) else self.compute_h(j, c, instances[i], instances)
            epsilon += self.D[i] * (1 if h_value != y_i else 0)
        return epsilon

    def compute_h(self, j, c, instance, instances):
        x_i = instance._feature_vector.feature_vector
        candidates = self.create_candidate_dict()
        if x_i.has_key(j):
            if x_i[j] > c:
                for instance in instances:
                    x_i_prime = instance._feature_vector.feature_vector
                    y_i_prime = instance._label.label
                    if x_i_prime.has_key(j) and x_i_prime[j] > c:
                        candidates[y_i_prime] += 1
            else:
                for instance in instances:
                    x_i_prime = instance._feature_vector.feature_vector
                    y_i_prime = instance._label.label
                    if x_i_prime.has_key(j) and x_i_prime[j] <= c:
                        candidates[y_i_prime] += 1
        else:
            for instance in instances:
                x_i_prime = instance._feature_vector.feature_vector
                y_i_prime = instance._label.label
                if x_i_prime.has_key(j) and x_i_prime[j] <= c:
                    candidates[y_i_prime] += 1
        candidates = sorted(candidates.items(), key=lambda tup: tup[1], reverse=True)
        self.h_cache[(j, c, instance)] = candidates[0][0]
        return candidates[0][0]

    def compute_z(self, a_t, instances, j, c):
        z = 0.0
        for i in range(len(instances)):
            x_i = instances[i]._feature_vector.feature_vector
            y_i = 1 if instances[i]._label.label == 1 else -1
            h_value = self.h_cache[(j, c, instances[i])] if self.h_cache.has_key((j, c, instances[i])) else self.compute_h(j, c, instances[i], instances)
            z += (self.D[i] * exp(-a_t * y_i * h_value))
        return z

    def predict(self, instance):
        candidates = self.create_candidate_dict()
        for t in range(len(self.h_t_list)):
            a_t = self.a_list[t]
            j_t, c_t = self.h_t_list[t]
            h_val = self.h_cache[(j_t, c_t, instance)] if self.h_cache.has_key((j_t, c_t, instance)) else self.compute_h(j_t, j_c, instance, self.instances)
            candidates[h_val] += a_t
        candidates = sorted(candidates.items(), key=lambda tup: tup[1], reverse=True)
        return candidates[0][0]

    #########################################################
    #  Helper functions -- probably can't do these wrong??? #
    #########################################################
    
    def initialize(self, instances):
        self.make_labels_list(instances)
        self.make_features_list(instances)
        self.make_hypothesis_list(instances)
        self.set_first_D(instances)
        self.instances = instances

    def make_labels_list(self, instances):
        for instance in instances:
            if instance._label.label not in self.all_labels:
                self.all_labels.append(instance._label.label) 

    def make_features_list(self, instances):
        for instance in instances:
            for feature in instance._feature_vector.feature_vector:
                if feature not in self.all_features:
                    self.all_features.append(feature)

    def make_hypothesis_list(self, instances):
        for j in self.all_features:
            all_dim_j_values = self.get_all_dim_j_values(instances, j)
            for c in all_dim_j_values:
                self.hypothesis_list.append((j, c))

    def set_first_D(self, instances):
        n = len(instances)
        for i in range(n):
            self.D.append(1.0 / n)

    def get_all_dim_j_values(self, instances, j):
        all_dim_j_values = []
        for instance in instances:
            x_i = instance._feature_vector.feature_vector
            if x_i.has_key(j):
                all_dim_j_values.append(x_i[j])
        all_dim_j_values.sort()
        return all_dim_j_values

    def create_candidate_dict(self):
        candidates = {}
        for label in self.all_labels:
            candidates[label] = 0
        return candidates

