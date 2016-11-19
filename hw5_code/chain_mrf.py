import numpy as np

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
        if x_i == 1 or x_i == self.n:
            temp = np.multiply(self.front_messages[x_i - 1], self.back_messages[x_i - 1])
            temp = temp / float(np.sum(temp))
            result = np.zeros(self.k + 1)
            result[1 : len(result)] = temp
        else:
            temp = np.multiply(self.front_messages[x_i - 1], self.back_messages[x_i - 1])
            temp = np.multiply(temp, self.call_unary_potential(x_i))
            temp = temp / float(np.sum(temp))
            result = np.zeros(self.k + 1)
            result[1 : len(result)] = temp
        return result

class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
        return 0.0
