from cs475_types import Predictor, ClassificationLabel
from math import sqrt, log, exp, pi

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
            known_instance_value = known_instance_features[feature] \
                    if known_instance_features.has_key(feature) else 0
            test_instance_value = test_instance_features[feature] \
                    if test_instance_features.has_key(feature) else 0
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
        self.K = 0
        self.r_vector = {}

    def initialize(self, instances):
        self.instances = instances
        self.N = len(self.instances)
        self.note_all_features()
        self.set_first_mu()
        self.set_default_lambda()
        self.set_rnk()

    def train(self, instances):
        self.initialize(instances)
        for t in range(self.T):
            self.E_step()
            self.M_step()

    def E_step(self):
        for i in range(self.N):
            x_i = self.instances[i]._feature_vector.feature_vector
            min_k, min_distance = -1, float('inf')
            for k in range(self.K):
                curr_distance = self.distance(self.mu_vector[k], x_i)
                if curr_distance < min_distance and curr_distance <= self.cluster_lambda:
                    min_k, min_distance = k, curr_distance
                elif curr_distance == min_distance and curr_distance <= self.cluster_lambda and k < min_k:
                    min_k = k # Break ties.
            if min_distance > self.cluster_lambda:
                # Start a new cluster and increment K afterwards.
                self.r_vector[i] = self.K
                self.mu_vector[self.K] = x_i
                self.K += 1
            else:
                self.r_vector[i] = min_k

    def M_step(self):
        for k in range(self.K):
            sum_rnk = self.sum_up_rnk(k)
            if sum_rnk == 0:
                new_mu_k = self.set_empty_prototype()
            else:
                new_mu_k = self.compute_new_mu_k(k, sum_rnk)
            self.mu_vector[k] = new_mu_k

    def predict(self, instance):
        # Assign example to the closest cluster.
        # This does not create any new clusters. 
        # i.e. no change in K!
        min_k, min_distance = -1, float('inf')
        for k in range(self.K):
            curr_distance = self.distance(self.mu_vector[k], instance._feature_vector.feature_vector)
            if curr_distance < min_distance:
                min_k, min_distance = k, curr_distance
            elif curr_distance == min_distance and k < min_k:
                min_k = k
        return min_k

    #####################################################
    # M-step() helper functions.
    #####################################################

    def compute_new_mu_k(self, k, sum_rnk):
        new_mu_k = self.set_empty_prototype()
        for i in range(self.N):
            if self.r_vector[i] == k:
                feature_vector_to_sum = self.instances[i]._feature_vector.feature_vector
                for j in feature_vector_to_sum.keys():
                    new_mu_k[j] += feature_vector_to_sum[j] / sum_rnk
        return new_mu_k

    def sum_up_rnk(self, k):
        sum_rnk = 0.0
        for i, cluster_number in self.r_vector.items():
            if cluster_number == k:
                sum_rnk += 1
        return float(sum_rnk)

    #####################################################
    # Initialization helper functions.
    #####################################################

    def note_all_features(self):
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector # type: dict(feature, value)
            for j in x_i.keys():
                if j not in self.all_features:
                    self.all_features.append(j)

    def set_first_mu(self):
        first_mu = self.set_empty_prototype()
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector
            for j in x_i.keys():
                first_mu[j] += x_i[j]
        for j in first_mu.keys():
            first_mu[j] /= float(self.N)
        self.mu_vector[0] = first_mu
        self.K += 1

    def set_default_lambda(self):
        if self.cluster_lambda == 0:
            self.cluster_lambda = 0.0
            first_mu = self.mu_vector[0]
            for i in range(self.N):
                x_i = self.instances[i]._feature_vector.feature_vector
                self.cluster_lambda += self.distance(x_i, first_mu)
            self.cluster_lambda /= float(self.N)

    def set_rnk(self):
        for i in range(self.N):
            self.r_vector[i] = -1

    #####################################################
    # Miscellaneous helper functions.
    #####################################################

    def distance(self, v1, v2):
        # Actually returns euclidian distance squared.
        # Did not use sqrt() because the algorithm doesn't
        # use the square rooted value ever.
        dist = 0.0
        for j in self.all_features:
            v1_value = v1[j] if v1.has_key(j) else 0.0
            v2_value = v2[j] if v2.has_key(j) else 0.0
            dist += (v1_value - v2_value)**2
        return dist

    def set_empty_prototype(self):
        new_prototpye = {}
        for j in self.all_features:
            new_prototpye[j] = 0.0
        return new_prototpye

       
""" HW4 """
class NaiveBayes(Predictor):

    def __init__(self, num_clusters, clustering_training_iterations):
        self.K = num_clusters
        self.T = clustering_training_iterations
        self.N = 0.0
        self.S = {}
        self.instances = []
        self.all_features = []
        self.clusters = {}
        self.mu_vector = {}
        self.sigma_vector = {}
        self.phi_vector = {}

    def initialize(self, instances):
        self.instances = instances
        self.N = len(self.instances)
        self.note_all_features()
        self.split_data()
        self.init_mu_k()
        self.compute_S()
        self.init_sigma_k()
        self.init_phi_k()

    def train(self, instances):
        self.initialize(instances)
        # print("Sj: " + str(self.S))
        # self.print_cluster_details(-1)
        for t in range(self.T):
            self.E_step()
            self.M_step()
            # print("\nAfter M-step: ")
            # self.print_cluster_details(9)

    def E_step(self):
        for i in range(self.N):
            x_i = self.instances[i]._feature_vector.feature_vector
            max_k = self.argmax_k(x_i)
            curr_k = self.find_current_cluster(i, x_i)
            self.update_cluster_assignment(i, x_i, curr_k, max_k)

    def M_step(self):
        for k in range(self.K):
            N_k = float(len(self.clusters[k]))
            self.update_phi_k(k, N_k)
            self.update_mu_k(k, N_k)
            self.update_sigma_k(k, N_k)

    def predict(self, instance):
        return self.argmax_k(instance._feature_vector.feature_vector)

    #####################################################
    # E-step() helper functions.
    #####################################################

    def argmax_k(self, x_i):
        max_k, max_value = -1, -float('inf')
        for k in range(self.K):
            value = self.evaluate_argmax_expression(k, x_i)
            if value > max_value:
                max_value, max_k = value, k
            elif value == max_value and k < max_k: # break ties.
                max_k = k
        return max_k

    def find_current_cluster(self, i, x_i):
        for k, cluster_k in self.clusters.items():
            if (i, x_i) in cluster_k:
                return k
        return -1

    def update_cluster_assignment(self, i, x_i, curr_k, max_k):
        if (i, x_i) not in self.clusters[max_k]:
            # print("===========================")
            # print((i, x_i))
            # print("-   -   -   -   -   -   - ")
            # print(self.clusters[curr_k])
            # print(self.clusters[max_k])
            self.clusters[curr_k].remove((i, x_i))
            # print("-   -   -   -   -   -   - ")            
            self.clusters[max_k].append((i, x_i))
            # print(self.clusters[curr_k])
            # print(self.clusters[max_k])
            # print("===========================")
            # print("")

    def evaluate_argmax_expression(self, k, x_i):
        first_part = log(self.phi_vector[k])
        # first_part = log(self.phi_vector[k]) if self.phi_vector[k] != 0 else -float('inf')
        # if first_part == 0:
        #     return 0.0
        second_part = 0.0
        mu_k, sigma_k = self.mu_vector[k], self.sigma_vector[k]
        for j in self.all_features:
            x_ij = x_i[j] if x_i.has_key(j) else 0.0
            logp = self.compute_normal(x_ij, mu_k[j], sigma_k[j])
            second_part += logp
            # if p == 0.0:
            #     return 0.0
            # else:
            #     second_part += log(p)
        return first_part + second_part

    def compute_normal(self, x_ij, mu_kj, sigma_kj):
        return log(1 / sqrt(2 * pi * sigma_kj)) - ((x_ij - mu_kj)**2) / (2 * sigma_kj)
        # if sigma_kj == 0:
        #     return 1.0 if x_ij >= mu_kj else 0.0
        # return (1.0 / sqrt(2 * pi * sigma_kj)) * exp(-((x_ij - mu_kj)**2) / (2 * sigma_kj))

    #####################################################
    # M-step() helper functions.
    #####################################################

    def update_phi_k(self, k, N_k):
        self.phi_vector[k] = (N_k + 1) / float(self.N + self.K)
    
    def update_mu_k(self, k, N_k):
        new_mu_k = self.set_empty_prototype()
        if N_k != 0:
            for i, x_i in self.clusters[k]:
                for j in self.all_features:
                    x_ij = x_i[j] if x_i.has_key(j) else 0.0
                    new_mu_k[j] += x_ij / N_k
        self.mu_vector[k] = new_mu_k

    def update_sigma_k(self, k, N_k):
        mu_k, new_sigma_k = self.mu_vector[k], self.set_empty_prototype()
        if N_k > 1:
            for i, x_i in self.clusters[k]:
                for j in self.all_features:
                    x_ij = x_i[j] if x_i.has_key(j) else 0.0
                    new_sigma_k[j] += (x_ij - mu_k[j])**2 / (N_k - 1)
                    # print("[" + str(x_ij) + " - " + str(mu_k[j]) + "]^2")
            for j in self.all_features:
                if new_sigma_k[j] == 0 or new_sigma_k[j] < self.S[j]:
                    new_sigma_k[j] = self.S[j]
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.sigma_vector[k] = new_sigma_k
            # print(N_k)
            # print(new_sigma_k)
            # print(self.clusters[2])
        else:
            self.sigma_vector[k] = self.S

    #####################################################
    # Initialization helper functions.
    #####################################################

    def note_all_features(self):
        for instance in self.instances:
            x_i = instance._feature_vector.feature_vector
            for j in x_i.keys():
                if j not in self.all_features:
                    self.all_features.append(j)

    def split_data(self):
        # Init empty clusters with empty lists.
        for k in range(self.K):
            self.clusters[k] = []

        # Divide data into K folds.
        for i in range(self.N):
            self.clusters[(i) % self.K].append((i, self.instances[i]._feature_vector.feature_vector))

    def init_mu_k(self):
        for k in range(self.K):
            N_k = float(len(self.clusters[k]))
            mu_k = self.set_empty_prototype()       
            for i, x_i in self.clusters[k]:
                for j, x_ij in x_i.items():
                    mu_k[j] += x_ij / N_k
            self.mu_vector[k] = mu_k

    def compute_S(self):
        mu_N = self.set_empty_prototype()
        for i in range(self.N):
            x_i = self.instances[i]._feature_vector.feature_vector
            for j, x_ij in x_i.items():
                mu_N[j] += x_ij / float(self.N)

        self.S = self.set_empty_prototype()
        for i in range(self.N):
            x_i = self.instances[i]._feature_vector.feature_vector
            for j in self.all_features:
                x_ij = x_i[j] if x_i.has_key(j) else 0.0
                self.S[j] += (0.01 * (x_ij - mu_N[j])**2 / float(self.N - 1))

    def init_sigma_k(self):
        for k in range(self.K):
            # print("==========================")
            # print(self.clusters[k])
            N_k = float(len(self.clusters[k]))
            mu_k, sigma_k = self.mu_vector[k], self.set_empty_prototype()
            if N_k > 1:
                for i, x_i in self.clusters[k]:
                    for j in self.all_features:
                        x_ij = x_i[j] if x_i.has_key(j) else 0.0
                        sigma_k[j] += (x_ij - mu_k[j])**2 / (N_k - 1)
                    for j in self.all_features:
                        if sigma_k[j] == 0 or sigma_k[j] < self.S[j]:
                            sigma_k[j] = self.S[j]
                self.sigma_vector[k] = sigma_k
            else:
                self.sigma_vector[k] = self.S

    def init_phi_k(self):
        for k in range(self.K):
            self.phi_vector[k] = (len(self.clusters[k]) + 1) / float(self.N + self.K)

    #####################################################
    # Miscellaneous helper functions.
    #####################################################

    def set_empty_prototype(self):
        new_prototpye = {}
        for j in self.all_features:
            new_prototpye[j] = 0.0
        return new_prototpye

    def print_cluster_details(self, t):
        print("============== t = " + str(t) + " ==============")
        print("")
        for key, value in self.clusters.items():
            print("------------- Cluster " + str(key) + " -------------")
            print("size: " + str(len(self.clusters[key])))
            print("means: " + str(self.mu_vector[key]))
            print("variances: " + str(self.sigma_vector[key]))
            print("probability: " + str(self.phi_vector[key]))
            # print("")
            # print("cluster: " + str(self.clusters[key]))
        print("")
        # print("")