
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
            if t == 0 and epsilon == 0:
                self.h_t_list.append((j, c))
                self.a_list.append(1)
                break
            a_t = 0.5 * log((1 - epsilon) / epsilon)
            if a_t < 0.000001:
                break
            self.h_t_list.append((j, c))
            self.a_list.append(a_t)
            for i in range(len(instances)):
                x_i = instances[i]._feature_vector.feature_vector
                y_i = 1 if instances[i]._label.label == 1 else -1
                h_val = self.h_cache[(j, c, instances[i])] if self.h_cache.has_key((j, c, instances[i])) \
                                                           else self.compute_h(j, c, instances[i], instances)
                z_val = self.z_cache[(a_t, j, c)] if self.z_cache.has_key((a_t, j, c)) \
                                                  else self.compute_z(a_t, instances, j, c)
                self.D[i] *= ((1.0 / z_val) * exp(-a_t * y_i * h_val))

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
            h_val = self.h_cache[(j, c, instances[i])] if self.h_cache.has_key((j, c, instances[i])) \
                                                       else self.compute_h(j, c, instances[i], instances)
            epsilon += self.D[i] * (1 if h_val != y_i else 0)
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
            h_val = self.h_cache[(j, c, instances[i])] if self.h_cache.has_key((j, c, instances[i])) \
                                                       else self.compute_h(j, c, instances[i], instances)    
            z += (self.D[i] * exp(-a_t * y_i * h_val))
        return z

    def predict(self, instance):
        candidates = self.create_candidate_dict()
        for t in range(len(self.h_t_list)):
            a_t = self.a_list[t]
            j_t, c_t = self.h_t_list[t]
            h_val = self.h_cache[(j_t, c_t, instance)] if self.h_cache.has_key((j_t, c_t, instance)) \
                                                       else self.compute_h(j_t, c_t, instance, self.instances)                
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
            for i in range(len(all_dim_j_values) - 1):
                c = (all_dim_j_values[i] + all_dim_j_values[i + 1]) / 2.0
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
                if x_i[j] not in all_dim_j_values:
                    all_dim_j_values.append(x_i[j])
        all_dim_j_values.sort()
        return all_dim_j_values

    def create_candidate_dict(self):
        candidates = {}
        for label in self.all_labels:
            candidates[label] = 0
        return candidates

