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
        self.create_all_dim_list()
        self.create_hypothesis_set()
        self.initialize_weights(len(instances))
        self.create_all_labels_list()
        for t in range(self.T):
            j_t, c_t = self.find_h_t()
            e_t = self.compute_error(j_t, c_t)
            if t == 0 and e_t == 0:
                self.h_t_list.append((j_t, c_t))
                self.a_t_list.append(1.0)
                break
            a_t = 0.5 * log((1 - e_t) / (e_t))
            if a_t < 0.000001:
                break
            self.h_t_list.append((j_t, c_t))
            self.a_t_list.append(a_t)
            self.update_weight(a_t, j_t, c_t)

    def predict(self, instance):
        candidates = self.create_candidates_map()
        x_i = instance._feature_vector.feature_vector
        y_i = 1.0 if instance._label.label == 1 else -1.0
        for t in range(len(self.h_t_list)):
            j = self.h_t_list[t][0]
            c = self.h_t_list[t][1]
            candidates[y_i] += (self.a_t_list[t] * (1.0 if self.compute_h(j, c, instance) == y_i else 0.0))
        candidates = sorted(candidates.items(), key=lambda tup: tup[1], reverse=True)
        return candidates[0][0]

    def update_weight(self, a, j, c):
        z_t = self.compute_z(a, j, c)
        for i in range(len(self.instances)):
            instance = self.instances[i]
            x_i = instance._feature_vector.feature_vector
            y_i = 1.0 if instance._label.label == 1 else -1.0
            self.D[i] *= ((1.0 / z_t) * exp(-a * y_i * self.compute_h(j, c, instance)))

    def compute_z(self, a, j, c):
        if self.z_value_cache.has_key((j,c,a)):
            return self.z_value_cache[(j,c,a)]
        else:
            z = 0.0
            for i in range(len(self.instances)):
                instance = self.instances[i]
                x_i = instance._feature_vector.feature_vector
                y_i = 1.0 if instance._label.label == 1 else -1.0
                z += (self.D[i] * exp(-a*y_i*self.compute_h(j, c, instance)))
            self.z_value_cache[(j,c,a)] = z
            return z

    def initialize_weights(self, n):
        for i in range(n):
            self.D.append(1.0 / n)

    def create_all_dim_list(self):
        for instance in self.instances:
            for feature in instance._feature_vector.feature_vector.keys():
                if feature not in self.all_dim_list:
                    self.all_dim_list.append(feature)
        self.all_dim_list.sort()

    def create_hypothesis_set(self):
        for j in self.all_dim_list:
            c_list = self.create_c_list_for_dim_j(j)
            for i in range(len(c_list) - 1):
                self.hypothesis_set.append((j, (c_list[i] + c_list[i + 1]) / 2.0))

    def find_h_t(self):
        min_error, min_j, min_c = float('inf'), 0, 0
        for j, c in self.hypothesis_set:
            error = self.compute_error(j, c)
            if error < min_error:
                min_error, min_j, min_c = error, j, c
        return min_j, min_c

    def compute_error(self, j, c):
        error = 0.0
        for i in range(len(self.instances)):
            instance = self.instances[i]
            x_i = instance._feature_vector.feature_vector
            y_i = 1.0 if instance._label.label == 1 else -1.0
            error += self.D[i] * (1 if self.compute_h(j, c, instance) == y_i else 0)
        return error

    def create_all_labels_list(self):
        for instance in self.instances:
            label = 1.0 if instance._label.label == 1 else -1.0
            if label not in self.all_labels:
                self.all_labels.append(label)

    def create_candidates_map(self):
        candidates = {}
        for label in self.all_labels:
            candidates[label] = 0.0
        return candidates

    def compute_h(self, j, c, instance):
        x_i = instance._feature_vector.feature_vector
        if self.h_value_cache.has_key((j,c,instance)):
            return self.h_value_cache[(j,c,instance)]
        else:
            candidates = self.create_candidates_map()
            if x_i.has_key(j) and x_i[j] > c:
                for instance in self.instances:
                    if instance._feature_vector.feature_vector.has_key(j) and instance._feature_vector.feature_vector[j] > c:
                        candidates[1.0 if instance._label.label == 1 else -1.0] += 1.0
            else:
                for instance in self.instances:
                    if instance._feature_vector.feature_vector.has_key(j) and instance._feature_vector.feature_vector[j] <= c:
                        candidates[1.0 if instance._label.label == 1 else -1.0] += 1.0
            candidates = sorted(candidates.items(), key=lambda tup:tup[1], reverse=True)
            self.h_value_cache[(j,c,instance)] = candidates[0][0]
            return candidates[0][0]

    def create_c_list_for_dim_j(self, j):
        c_list = []
        for instance in self.instances:
            for feature, value in instance._feature_vector.feature_vector.items():
                if feature == j and value not in c_list:
                    c_list.append(value)
        c_list.sort()
        return c_list
