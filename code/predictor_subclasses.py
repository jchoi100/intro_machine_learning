from cs475_types import Predictor, ClassificationLabel
from math import sqrt, log, exp

""" HW1 """
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

""" HW1 """
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

""" HW2 """
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

""" HW2 """
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

""" HW3 """
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

""" HW3 """
class AdaBoost(Predictor):

    def __init__(self, num_boosting_iterations):
        self.T = num_boosting_iterations
        self.D = []
        self.all_labels = []
        self.all_features = []
        self.h_t_list = []
        self.a_t_list = []
        self.hypothesis_list = []
        self.instances = []

    def train(self, instances):
        self.initialize(instances)
        for t in range(self.T):
            j, c = self.get_h_t()
            e_t = self.compute_epsilon(j, c)
            if t == 0 and e_t < 0.000001:
                self.h_t_list.append((j, c))
                self.a_t_list.append(1.0)
                break
            if e_t < 0.000001:
                break
            a_t = 0.5 * log((1 - e_t) / e_t)
            self.h_t_list.append((j, c))
            self.a_t_list.append(a_t)
            self.update_weights(j, c, a_t)

    def update_weights(self, j, c, a_t):
        for i in range(len(self.instances)):
            instance = self.instances[i]
            y_i = 1.0 if instance._label.label == 1 else -1.0
            h_val = self.compute_h(j, c, instance)
            z_val = self.compute_z(j, c, a_t)
            self.D[i] *= ((1.0 / z_val) * exp(-a_t * y_i * h_val))

    def get_h_t(self):
        min_epsilon = float('inf')
        min_j, min_c = -1, -1
        for j, c in self.hypothesis_list:
            curr_epsilon = self.compute_epsilon(j, c)
            if curr_epsilon < min_epsilon:
                min_epsilon = curr_epsilon
                min_j, min_c = j, c
        return min_j, min_c

    def compute_epsilon(self, j, c):
        epsilon = 0.0
        for i in range(len(self.instances)):
            instance = self.instances[i]
            y_i = 1.0 if instance._label.label == 1 else -1.0
            h_val = self.compute_h(j, c, instance)
            epsilon += self.D[i] * (1.0 if h_val != y_i else 0.0)
        return epsilon

    def compute_h(self, j, c, instance):
        x_i = instance._feature_vector.feature_vector
        candidates = self.create_candidate_dict()
        if x_i.has_key(j) and x_i[j] > c:
            for instance in self.instances:
                x_i_prime = instance._feature_vector.feature_vector
                y_i_prime = 1.0 if instance._label.label == 1 else -1.0
                if x_i_prime.has_key(j) and x_i_prime[j] > c:
                    candidates[y_i_prime] += 1
        else:
            for instance in self.instances:
                x_i_prime = instance._feature_vector.feature_vector
                y_i_prime = 1.0 if instance._label.label == 1 else -1.0
                if x_i_prime.has_key(j) and x_i_prime[j] <= c:
                    candidates[y_i_prime] += 1
        candidates = sorted(candidates.items(), key=lambda tup: tup[1], reverse=True)
        return candidates[0][0]

    def compute_z(self, j, c, a_t):
        z = 0.0
        for i in range(len(self.instances)):
            instance = self.instances[i]
            y_i = 1.0 if instance._label.label == 1 else -1.0
            h_val = self.compute_h(j, c, instance)    
            z += (self.D[i] * exp(-a_t * y_i * h_val))
        return z

    def predict(self, instance):
        candidates = self.create_candidate_dict()
        for t in range(len(self.h_t_list)):
            a_t = self.a_t_list[t]
            j_t, c_t = self.h_t_list[t]
            h_val = self.compute_h(j_t, c_t, instance)
            candidates[h_val] += a_t
        candidates = sorted(candidates.items(), key=lambda tup: tup[1], reverse=True)
        return 1 if candidates[0][0] == 1 else 0

    #########################################################
    #  Helper functions                                     #
    #########################################################
    
    def initialize(self, instances):
        self.instances = instances
        self.make_labels_list()
        self.make_features_list()
        self.make_hypothesis_list()
        self.set_first_D()

    def make_labels_list(self):
        for instance in self.instances:
            label = 1.0 if instance._label.label == 1 else -1.0
            if label not in self.all_labels:
                self.all_labels.append(label)
        self.all_labels.sort()

    def make_features_list(self):
        for instance in self.instances:
            for feature in instance._feature_vector.feature_vector.keys():
                if feature not in self.all_features:
                    self.all_features.append(feature)
        self.all_features.sort()

    def make_hypothesis_list(self):
        for j in self.all_features:
            all_dim_j_values = self.get_all_dim_j_values(j)
            for i in range(len(all_dim_j_values) - 1):
                c = (all_dim_j_values[i] + all_dim_j_values[i + 1]) / 2.0
                self.hypothesis_list.append((j, c))

    def set_first_D(self):
        n = len(self.instances)
        for i in range(n):
            self.D.append(1.0 / n)

    def get_all_dim_j_values(self, j):
        all_dim_j_values = []
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector
            if x_i.has_key(j):
                if x_i[j] not in all_dim_j_values:
                    all_dim_j_values.append(x_i[j])
        all_dim_j_values.sort()
        return all_dim_j_values

    def create_candidate_dict(self):
        candidates = {}
        for label in self.all_labels:
            candidates[label] = 0
        return candidates


""" HW4 """
class LambdaMeans(Predictor):

    def __init__(self, cluster_lambda, clustering_training_iterations):
        self.cluster_lambda = cluster_lambda
        self.T = clustering_training_iterations
        self.N = 0.0
        self.all_features = []
        self.instances = []
        self.mu_vector = {}
        self.K = 1
        self.r_vector = {}

    def train(self, instances):
        self.initialize(instances)
        for t in range(self.T):
            self.E_step()
            self.M_step()

    def E_step(self):
        for i in range(self.N):
            curr_instance = self.instances[i]
            min_cluster = 0
            min_distance = float('inf')
            for k in range(self.K):
                k_index = k + 1
                curr_prototype = self.mu_vector[k_index]
                curr_distance = self.compute_euclidian_distance(curr_prototype, curr_instance._feature_vector.feature_vector)
                if curr_distance < min_distance and curr_distance <= self.cluster_lambda:
                    min_cluster = k_index
                    min_distance = curr_distance
            if min_distance > self.cluster_lambda:
                self.K += 1
                self.r_vector[i] = self.K
                self.mu_vector[self.K] = curr_instance
            else:
                self.r_vector[i] = min_cluster

    def M_step(self):
        for k in range(self.K):
            k_index = k + 1
            sum_rnk = 0.0
            for instance_number, cluster_number in self.r_vector.items():
                if cluster_number == k_index:
                    sum_rnk += 1
            new_mu_k = self.set_empty_prototype()
            for i in range(self.N):
                if self.r_vector[i] == k_index:
                    feature_vector_to_sum = self.instances[i]._feature_vector.feature_vector
                    for feature, value in feature_vector_to_sum.items():
                        new_mu_k[feature] += value
            for key, value in new_mu_k.items():
                new_mu_k[key] /= sum_rnk
            self.mu_vector[k_index] = new_mu_k

    def set_empty_prototype(self):
        new_prototpye = {}
        for feature in self.all_features:
            new_prototpye[feature] = 0.0
        return new_prototpye

    def predict(self, instance):
        # Assign example to the closest cluster.
        # This does not create any new clusters. i.e. no change in K!
        x_i = instance._feature_vector.feature_vector
        min_cluster = 0
        min_distance = float('inf')
        for k in range(self.K):
            k_index = k + 1
            curr_prototype = self.mu_vector[k_index]
            curr_distance = self.compute_euclidian_distance(curr_prototype, instance._feature_vector.feature_vector)
            if curr_distance < min_distance:
                min_cluster = k_index
                min_distance = curr_distance
        return min_cluster

    def initialize(self, instances):
        self.instances = instances
        self.N = len(self.instances)

        # 1. Take note of all the features that appear in all the instances
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector # type: dict(feature, value)
            for feature, value in x_i.items():
                if feature not in self.all_features:
                    self.all_features.append(feature)

        # 2. Set mu_1
        self.set_first_mu()

        # 3. Check lambda value 0
        if self.cluster_lambda == 0:
            self.set_default_lambda()

        # 4. Initialize r_nk vector. r[ith instance] = cluster k = 0
        for i in range(self.N):
            self.r_vector[i] = 0

    def set_first_mu(self):
        first_mu = {}
        for feature in self.all_features:
            first_mu[feature] = 0.0
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector
            for feature, value in x_i.items():
                first_mu[feature] += value
        for feature in first_mu.keys():
            first_mu[feature] /= (self.N * 1.0)
        self.mu_vector[1] = first_mu

    def set_default_lambda(self):
        self.cluster_lambda = 0.0
        first_mu = self.mu_vector[1]
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector
            self.cluster_lambda += self.compute_euclidian_distance(x_i, first_mu)**2
        self.cluster_lambda /= (self.N * 1.0)

    def compute_euclidian_distance(self, v1, v2):
        dist = 0.0
        for feature in self.all_features:
            v1_value = v1[feature] if v1.has_key(feature) else 0.0
            v2_value = v2[feature] if v2.has_key(feature) else 0.0
            dist += (v1_value - v2_value)**2
        return sqrt(dist)


    """
    NOTES: train

    Do NOT use the labels in the training data during training!!!

    Learn the cluster parameters based on the training examples.
    The result of this method are the cluster parameters:
    the means mu_k and number of clusters K.
    """
    """
    NOTES: predict

    Since the clustering algorithm cannot read the correct labels,
    we must be clear about what "label" you are returning here!!

    This method should label the examples by assigning them to the
    closest cluster. The value of the label should be the cluster 
    index. (May use either 0-based indexing or 1-indexing for cluster
    index k) You should NOT increase the number of available clusters
    at prediction time.
    """

       
""" HW4 """
# class NaiveBayes(Predictor):

#     def train(self, instances):

#     def predict(self, instance):

#        