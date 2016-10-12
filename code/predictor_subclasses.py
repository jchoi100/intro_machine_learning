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
        self.knn = knn
        self.is_weighted = is_weighted
        self.instances = []
        self.all_features = []

    def train(self, instances):
        self.instances = instances
        for instance in instances:
            for feature in instance._feature_vector.feature_vector:
                if feature not in self.all_features:
                    self.all_features.append(feature)

    def predict(self, instance):
        neighbors = []
        for known_instance in self.instances:
            x_i = known_instance._feature_vector.feature_vector
            y_i = known_instance._label.label
            distance = self.compute_distance(x_i, instance._feature_vector.feature_vector)
            neighbors.append((y_i, distance))
        neighbors = sorted(neighbors, key=lambda tup: (tup[1], tup[0]))
        nearest_neighbors = neighbors[0:self.knn]
        votes = {}
        for neighbor in nearest_neighbors:
            if self.is_weighted:
                if votes.has_key(neighbor[0]):
                    votes[neighbor[0]] -= 1.0 / (1 + neighbor[1]**2)
                else:
                    votes[neighbor[0]] = -1.0 / (1 + neighbor[1]**2)
            else:
                if votes.has_key(neighbor[0]):
                    votes[neighbor[0]] -= 1
                else:
                    votes[neighbor[0]] = -1
        votes = sorted(votes.items(), key=lambda tup: (tup[1], tup[0]))
        return votes[0][0]

    def compute_distance(self, known_instance_features, test_instance_features):
        distance = 0
        for feature in self.all_features:
            known_instance_value = known_instance_features[feature] if known_instance_features.has_key(feature) else 0
            test_instance_value = test_instance_features[feature] if test_instance_features.has_key(feature) else 0
            distance += (known_instance_value - test_instance_value)**2
        return sqrt(distance)


class AdaBoost(Predictor):

    def __init__(self, num_boosting_iterations):
        self.T = num_boosting_iterations
        self.h_t_list = []
        self.a_t_list = []
        self.hypothesis_set = []
        self.D = []
        self.all_dim_list = []
        self.all_labels = []
        self.instances = []
        self.h_value_cache = {}
        self.z_value_cache = {}

    def train(self, instances):
        self.instances = instances
        self.create_all_dim_list(instances)
        self.create_hypothesis_set(instances)
        self.initialize_weights(len(instances))
        self.create_all_labels_list(instances)
        for t in range(self.T):
            j_t, c_t = self.find_h_t(instances)
            e_t = self.compute_error(j_t, c_t, instances)
            if e_t == 0:
                self.h_t_list.append((j_t, c_t))
                self.a_t_list.append(1)
                break
            a_t = 0.5 * log((1 - e_t) / (e_t))
            if a_t < 0.000001:
                break
            self.h_t_list.append((j_t, c_t))
            self.a_t_list.append(a_t)
            self.update_weight(a_t, j_t, c_t, instances)

    def predict(self, instance):
        candidates = self.create_candidates_map()
        x_i = instance._feature_vector.feature_vector
        y_i = 1 if instance._label.label == 1 else -1
        for t in range(self.T):
            j = self.h_t_list[t][0]
            c = self.h_t_list[t][1]
            candidates[y_i] += (self.a_t_list[t] * (1 if self.compute_h(j, c, x_i, self.instances) == y_i else 0))
        candidates = sorted(candidates.items(), key=lambda tup: tup[1], reverse=True)
        return candidates[0][0]

    def update_weight(self, a, j, c, instances):
        z_t = self.compute_z(a, j, c, instances)
        for i in range(len(instances)):
            x_i = instance._feature_vector.feature_vector
            y_i = 1 if instance._label.label == 1 else -1
            self.D[i] *= ((1.0 / z_t) * exp(-a*y_i*self.compute_h(j, c, x_i, instances)))

    def compute_z(self, a, j, c, instances):
        if self.z_value_cache.has_key((j,c,a)):
            return z_value_cache[(j,c,a)]
        else:
            z = 0.0
            for i in range(len(instances)):
                instance = instances[i]
                x_i = instance._feature_vector.feature_vector
                y_i = 1 if instance._label.label == 1 else -1
                z += (self.D[i] * exp(-a*y_i*self.compute_h(j, c, x_i, instances)))
            z_value_cache[(j,c,a)] = z
            return z

    def initialize_weights(self, n):
        for i in range(n):
            self.D.append(1.0 / n)

    def create_all_dim_list(self, instances):
        for instance in instances:
            for feature in instance._feature_vector.feature_vector.keys():
                if feature not in self.all_dim_list:
                    self.all_dim_list.append(feature)
        self.all_dim_list.sort()

    def create_hypothesis_set(self, instances):
        for j in self.all_dim_list:
            c_list = create_c_list_for_dim_j(instances, j)
            for i in range(len(c_list) - 1):
                self.hypothesis_set.append((j, (c_list[i] + c_list[i + 1]) / 2.0))

    def find_h_t(self, instances):
        min_error, min_j, min_c = float('inf'), 0, 0
        for j, c in self.hypothesis_set:
            error = self.compute_error(j, c, instances)
            if error < min_error:
                min_error, min_j, min_c = error, j, c
        return min_j, min_c

    def compute_error(self, j, c, instances):
        error = 0.0
        for i in range(len(instances)):
            instance = instances[i]
            x_i = instance._feature_vector.feature_vector
            y_i = 1 if instance._label.label == 1 else -1
            error += self.D[i] * (1 if self.compute_h(j, c, x_i, instances) == y_i else 0)
        return error

    def create_all_labels_list(self, instances):
        for instance in instances:
            label = 1 if instance._label.label == 1 else -1
            if label not in self.all_labels:
                self.all_labels.append(label)

    def create_candidates_map(self):
        candidates = {}
        for label in self.all_labels:
            candidates[label] = 0.0
        return candidates

    def compute_h(self, j, c, x_i, instances):
        if self.h_value_cache.has_key((j,c,x_i)):
            return self.h_value_cache[(j,c,x_i)]
        else:
            candidates = self.create_candidates_map(instances)
            if x_i.has_key(j) and x_i[j] > c:
                for instance in instances:
                    if instance._feature_vector.feature_vector.has_key(j) and instance._feature_vector.feature_vector[j] > c:
                        candidates[1 if instance._label.label == 1 else -1] += 1
            else:
                for instance in instances:
                    if instance._feature_vector.feature_vector.has_key(j) and instance._feature_vector.feature_vector[j] <= c:
                        candidates[1 if instance._label.label == 1 else -1] += 1
            candidates = sorted(candidates.items(), key=lambda tup:tup[1], reverse=True)
            self.h_value_cache[(j,c,x_i)] = candidates[0][0]
            return candidates[0][0]

    def create_c_list_for_dim_j(self, instances, j):
        c_list = []
        for instance in instances:
            for feature, value in instance._feature_vector.feature_vector.items():
                if feature == j and value not in c_list:
                    c_list.append(value)
        return c_list













