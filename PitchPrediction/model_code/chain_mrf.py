import math
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
        self.N = self._potentials.chain_length()
        self.K = self._potentials.num_x_values()
        self.Forward_f_to_x = {}
        self.Backward_f_to_x = {}
        for i in range(self.N):
            self.Forward_f_to_x.update({i+1 : [1.0]*(self.K+1)})
            self.Backward_f_to_x.update({i+1 : [1.0]*(self.K+1)})
        self.forwardStart()
        self.backwardStart()

    def forwardStart(self):
        # set base case
        f_to_x = self.Forward_f_to_x
        f_to_x.update({1 : [1.0]*(self.K+1)})

        # start recurse
        self.forwardRecurse(self.Unary_Potential_n(1), self.N+1)

    def forwardRecurse(self, message, n):
        unary_p = self._potentials._potentials1
        binary_p = self._potentials._potentials2

        f_sum = [0.0] * (self.K + 1)
        for var_from in range(1, self.K+1):
            for var_to in range(1, self.K+1):
                f_sum[var_from] += binary_p[n][var_to][var_from] * message[var_to]

        n = (n + 1 - self.N)
        self.Forward_f_to_x.update({n:f_sum})
        
        for k in range(1, self.K+1):
            message[k] = f_sum[k] * unary_p[n][k]

        if n < self.N:
            self.forwardRecurse(message, self.N + n)

    def backwardStart(self):
        # set base case
        f_to_x = self.Backward_f_to_x
        f_to_x.update({self.N : [1.0]*(self.K+1)})

        self.backwardRecurse(self.Unary_Potential_n(self.N), 2*self.N-1)
#        print 'back-edges:', f_to_x

    def backwardRecurse(self, message, n):
        unary_p = self._potentials._potentials1
        binary_p = self._potentials._potentials2

        # sum up the marginal potential
        f_sum = [0.0] * (self.K + 1)
        for var_from in range(1, self.K+1):
            for var_to in range(1, self.K+1):
                f_sum[var_from] += binary_p[n][var_from][var_to] * message[var_to]

        # update n back to unary
        n = (n - self.N)
        self.Backward_f_to_x.update({n:f_sum})
        
        # recurse back with updated messege
        for k in range(1, self.K+1):
            message[k] = f_sum[k] * unary_p[n][k]

        # until last one reached
        if n > 1:
            self.backwardRecurse(message, self.N + n - 1)

    def Unary_Potential_n(self, n):
        unary_p = self._potentials._potentials1
        f_x_n = [1.0]*(self.K+1)
        for k in range(1,self.K+1):
            f_x_n[k] = unary_p[n][k]
        return f_x_n

    def marginal_probability(self, x_i):
        unary_p = self._potentials._potentials1
        F = self.Forward_f_to_x
        B = self.Backward_f_to_x

        result = [0.0]*(self.K+1)
        Z = 0.0
        for k in range(1,self.K+1):
            result[k] = F.get(x_i)[k]*unary_p[x_i][k]*B.get(x_i)[k]
            Z += result[k]

        for k in range(1,self.K+1):
            result[k] = result[k] / Z
        return result

    def getZ(self, x_i):
        unary_p = self._potentials._potentials1
        F = self.Forward_f_to_x
        B = self.Backward_f_to_x
        Z = 0.0
        for k in range(1,self.K+1):
            Z += F.get(x_i)[k]*unary_p[x_i][k]*B.get(x_i)[k]
        return Z        

#####################################################################
# MAX SUM
#####################################################################

class MaxSum:
    def __init__(self, p):
        self.sp = SumProduct(p)
        self._potentials = p
        self._assignments = [None] * (p.chain_length() + 1)
        self.N = self._potentials.chain_length()
        self.K = self._potentials.num_x_values()
        self.F_M = {}
        self.B_M = {}

        self.forward = 1
        self.backward = -1

        for i in range(self.N):
            self.F_M.update({i+1 : [None]*(self.K+1)})
            self.B_M.update({i+1 : [None]*(self.K+1)})

        self.forwardStart()
        self.backwardStart()

    def forwardStart(self):
        self.F_M.update({1 : self.Log_Unary_Potential_n(1)})
        self.f_to_x(self.N + 1, self.Log_Unary_Potential_n(1), self.forward)

    def backwardStart(self):
        self.B_M.update({self.N : self.Log_Unary_Potential_n(self.N)})
        self.f_to_x(2*self.N - 1, self.Log_Unary_Potential_n(self.N), self.backward)

    def f_to_x(self, binary_index, message_sent, direction):
        binary_p = self._potentials._potentials2

        matrix = [[None for x in range(self.K+1)] for y in range(self.K+1)]
        max_assignment = [0.0] * (self.K+1)
        max_assignment_prob = [0.0] * (self.K+1)

        if direction is self.forward:                        
            for var_from in range(1,self.K+1):
                for var_to in range(1,self.K+1):
                    matrix[var_from][var_to] = math.log(binary_p[binary_index][var_from][var_to]) + message_sent[var_from]

            for var_to in range(self.K+1):
                maxprob = -np.inf
                argmax = None
                for var_from in range(1,self.K+1):
                    if matrix[var_from][var_to] > maxprob:
                        argmax = var_from
                        maxprob = matrix[var_from][var_to]
                max_assignment[var_to] = argmax
                max_assignment_prob[var_to] = maxprob

        else:
            for var_to in range(1,self.K+1):
                for var_from in range(1,self.K+1):
                    matrix[var_to][var_from] = math.log(binary_p[binary_index][var_from][var_to]) + message_sent[var_to]

            for var_to in range(self.K+1):
                maxprob = -np.inf
                argmax = None
                for var_from in range(1,self.K+1):
                    if matrix[var_from][var_to] > maxprob:
                        argmax = var_from
                        maxprob = matrix[var_from][var_to]
                max_assignment[var_to] = argmax
                max_assignment_prob[var_to] = maxprob

        next_message = max_assignment_prob
        if direction is self.forward:
            unary_index = binary_index - self.N + 1
            self.F_M.update({unary_index : next_message})
        else:
            unary_index = binary_index - self.N
            self.B_M.update({unary_index : next_message})            
        self.x_to_f(unary_index, next_message, direction)

    def x_to_f(self, unary_index, message_sent, direction):
        if unary_index >= self.N or unary_index <= 1:
            return 

        unary_p = self._potentials._potentials1
        next_message = self.elemSum(self.Log_Unary_Potential_n(unary_index), message_sent)

        if direction is self.forward:
            self.f_to_x(unary_index + self.N, next_message, direction)
        else:
            self.f_to_x(unary_index + self.N - 1, next_message, direction)

    def Unary_Potential_n(self, n):
        unary_p = self._potentials._potentials1
        f_x_n = [None]*(self.K+1)
        for k in range(1,self.K+1):
            f_x_n[k] = unary_p[n][k]
        return f_x_n

    def Log_Unary_Potential_n(self, n):
        unary_p = self._potentials._potentials1
        f_x_n = [None]*(self.K+1)
        for k in range(1,self.K+1):
            f_x_n[k] = math.log(unary_p[n][k])
        return f_x_n

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        maxarg_probs = [0.0] * (self.N+1)

        for n in range(1,self.N+1):
            if n==1:
                sum_vars = self.elemSum(self.Log_Unary_Potential_n(n), self.B_M.get(n))
            elif n==self.N:
                sum_vars = self.elemSum(self.Log_Unary_Potential_n(n), self.F_M.get(n))
            else:
                sum_vars = self.elemSum(self.elemSum(self.Log_Unary_Potential_n(n), self.B_M.get(n)), self.F_M.get(n))
            maxarg_probs[n] = max(sum_vars)
            self._assignments[n] = sum_vars.index(maxarg_probs[n])
        max_prob = sum(maxarg_probs)
        z = self.sp.getZ(x_i)
        return maxarg_probs[x_i] - math.log(z)

    def elemSum(self, unary, X):
        Z = [0.0]*(self.K+1)
        Z[0] = None
        for k in range(1,self.K+1):
            Z[k] = unary[k] + X[k]
        return Z            