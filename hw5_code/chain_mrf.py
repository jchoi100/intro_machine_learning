import numpy as np
from math import log

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self.n = p.chain_length()
        self.k = p.num_x_values()
        self.front_messages = []
        self.back_messages = []
        self.norm_constants = np.zeros(self.n + 1)
        self.front_pass()
        self.back_pass()
        self.back_messages = list(reversed(self.back_messages))

    def front_pass(self):
        prev_vector = np.ones(self.k)
        self.front_messages.append(self.call_unary_potential(1))
        for i in range(1, self.n):
            unary_potential = self.call_unary_potential(i)
            temp_message = np.multiply(prev_vector, unary_potential)
            binary_potential = self.call_binary_potential(i)
            prev_vector = np.dot(temp_message.T, binary_potential)
            self.front_messages.append(prev_vector)

    def back_pass(self):
        prev_vector = np.ones(self.k)
        self.back_messages.append(self.call_unary_potential(self.n))
        for i in range(self.n, 1, -1):
            unary_potential = self.call_unary_potential(i)
            temp_message = np.multiply(prev_vector, unary_potential)
            binary_potential = self.call_binary_potential(i - 1)
            prev_vector = np.dot(binary_potential, temp_message.T)
            self.back_messages.append(prev_vector)

    def call_unary_potential(self, i):
        potential = np.zeros(self.k)
        for j in range(1, self.k + 1):
            potential[j - 1] = self._potentials.potential(i, j)
        return potential

    def call_binary_potential(self, i):
        potential = np.zeros((self.k, self.k))
        for m in range(self.k):
            for n in range(self.k):
                potential[m][n] = self._potentials.potential(i + self.n, m + 1, n + 1)
        return potential

    def marginal_probability(self, x_i):
        temp = np.multiply(self.front_messages[x_i - 1], self.back_messages[x_i - 1])
        if x_i != 1 and x_i != self.n:
            temp = np.multiply(temp, self.call_unary_potential(x_i))
        self.norm_constants[x_i] = float(np.sum(temp))
        temp = temp / float(np.sum(temp))
        result = np.zeros(self.k + 1)
        result[1 : len(result)] = temp
        return result

class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [None] * (p.chain_length() + 1)
        self.sp = SumProduct(p)
        self.n = p.chain_length()
        self.k = p.num_x_values()
        for i in range(1, self.n + 1):
            self.sp.marginal_probability(i)
        self.norm_constants = self.sp.norm_constants
        self.unary_potential = self._potentials._potentials1
        self.binary_potential = self._potentials._potentials2
        self.front_messages = {1: self.call_unary_potential(1)}
        self.back_messages = {self.n: self.call_unary_potential(self.n)}
        self.front_pass(self.n + 1, self.call_unary_potential(1))
        self.back_pass(2 * self.n - 1, self.call_unary_potential(self.n))

    def front_pass(self, i, u):
        tab = np.zeros((self.k + 1, self.k + 1))
        next_u = np.zeros(self.k + 1)
        for s in range(1, self.k + 1):
            for t in range(1, self.k + 1):
                tab[s][t] = log(self.binary_potential[i][s][t]) + u[s]
        for t in range(self.k + 1):
            max_probability = -float('inf')
            for s in range(1, self.k + 1):
                if tab[s][t] > max_probability:
                    max_probability = tab[s][t]
            next_u[t] = max_probability
        next_i = i - self.n + 1
        self.front_messages[next_i] = next_u
        self.front_x_f(next_i, next_u)

    def back_pass(self, i, u):
        tab = np.zeros((self.k + 1, self.k + 1))
        next_u = np.zeros(self.k + 1)
        for t in range(1, self.k + 1):
            for s in range(1, self.k + 1):
                tab[t][s] = log(self.binary_potential[i][s][t]) + u[t]
        for t in range(1, self.k + 1):
            max_probability = -float('inf')
            for s in range(1, self.k + 1):
                if tab[s][t] > max_probability:
                    max_probability = tab[s][t]
            next_u[t] = max_probability
        next_i = i - self.n
        self.back_messages[next_i] = next_u
        self.back_x_f(next_i, next_u)

    def front_x_f(self, i, u):
        if i >= self.n:
            return
        next_u = self.vector_sum(self.call_unary_potential(i), u)
        self.front_pass(i + self.n, next_u)

    def back_x_f(self, i, u):
        if i <= 1:
            return
        next_u = self.vector_sum(self.call_unary_potential(i), u)
        self.back_pass(i + self.n - 1, next_u)

    def call_unary_potential(self, x_i):
        potential = np.zeros(self.k + 1)
        for i in range(1, self.k + 1):
            potential[i] = log(self.unary_potential[x_i][i])
        return potential

    def get_assignments(self):
        return self._assignments

    def vector_sum(self, a, b, c=None):
        if c is None:
            c = np.zeros(self.k + 1)
        result = np.zeros(self.k + 1)
        result[0] = -float('inf')
        for i in range(1, self.k + 1):
            result[i] = a[i] + b[i] + c[i]
        return result

    def max_probability(self, x_i):
        max_p = np.zeros(self.n + 1)
        for i in range(1, self.n + 1):
            if i == 1:
                agg = self.vector_sum(self.call_unary_potential(i), self.back_messages[i])
            elif i == self.n:
                agg = self.vector_sum(self.call_unary_potential(i), self.front_messages[i])
            else:
                agg = self.vector_sum(self.call_unary_potential(i), self.front_messages[i], self.back_messages[i])
            max_p[i] = np.amax(agg)
            self._assignments[i] = np.where(agg==max_p[i])[0][0]
        return max_p[x_i] - log(self.norm_constants[x_i])
