# Author: SangHyeon (Alex) Ahn
import math
from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

class AdaBoost(Predictor):
    def __init__(self, T):
        self.T = T
        self.alpha = [0]*T
        self.D = []
        self.H = [] # [j_dim, h_t_c, h_t_label]
        self.error_max = 0.000001

    # Training
    def train(self, instances):
        n = len(instances)
        self.D = [1.0/n]*(n)
        j_curr = 0
        j_max = 0
        for instance in instances:
            try:
                j_curr = max(instance.get_feature_vector().getKeys(), key=int)
                if j_max < j_curr:
                    j_max = j_curr
            except:
                print('no elements')

        for t in range(self.T):
            j_hypo_instance_index = [0]*j_max
            j_hypo_c = [0]*j_max
            j_hypo_error = [0]*j_max
            j_hypo_label = [0]*j_max

            for j in range(j_max):
                i_classifier_corrects = [0]*(n)
                i_classifier_errors = [0]*(n)
                i_classifier_labels = [0]*(n)

                for i in range(n):
                    c_i = instances[i].get_feature_vector().get(j+1)
                    i_classifier_corrects_sum = 0
                    i_classifier_corrects_sum_at_one = 0
                    i_classifier_corrects_sum_at_zero = 0
                    i_classifier_errors_sum = 0
                    i_classifier_errors_sum_at_one = 0
                    i_classifier_errors_sum_at_zero = 0
                    for k in range(n):
                        y_k = instances[k].get_label().get_int_label()
                        x_kj = instances[k].get_feature_vector().get(j+1)
                        if (x_kj > c_i) and (y_k is 1):
                            i_classifier_corrects_sum_at_one += 1
                            i_classifier_errors_sum_at_zero += self.D[k]
                        elif (x_kj <= c_i) and (y_k is 0):
                            i_classifier_corrects_sum_at_one += 1
                            i_classifier_errors_sum_at_zero += self.D[k]
                        elif (x_kj > c_i) and (y_k is 0):
                            i_classifier_corrects_sum_at_zero += 1
                            i_classifier_errors_sum_at_one += self.D[k]
                        elif (x_kj <= c_i) and (y_k is 1):
                            i_classifier_corrects_sum_at_zero += 1
                            i_classifier_errors_sum_at_one += self.D[k]
                        # there is no other case
                        else:
                            print('Error')
                            break #ERROR

                    if i_classifier_corrects_sum_at_one > i_classifier_corrects_sum_at_zero:
                        i_classifier_label = 1
                        i_classifier_corrects_sum = i_classifier_corrects_sum_at_one
                        i_classifier_errors_sum = i_classifier_errors_sum_at_one
                    else:
                        i_classifier_label = 0
                        i_classifier_corrects_sum = i_classifier_corrects_sum_at_zero
                        i_classifier_errors_sum = i_classifier_errors_sum_at_zero

                    # store decision stump calc result.
                    i_classifier_corrects[i] = [i, i_classifier_corrects_sum]
                    i_classifier_errors[i] = i_classifier_errors_sum
                    i_classifier_labels[i] = i_classifier_label

                # select best h_j_c by separation
                i_classifier_corrects.sort(key = lambda tup:tup[1], reverse = True)
                h_j_c = i_classifier_corrects.pop(0)
                h_j_c_index_i = h_j_c[0]

                # store h_j_c info
                j_hypo_c[j] = instances[h_j_c_index_i].get_feature_vector().get(j+1)
                j_hypo_error[j] = [j, i_classifier_errors[h_j_c_index_i]]
                j_hypo_label[j] = i_classifier_labels[h_j_c_index_i]

            # select h_t (minimum error h_j_t)
            j_hypo_error.sort(key = lambda tup:tup[1])
            h_t_choice = j_hypo_error.pop(0)

            feature_dim_j = h_t_choice[0]
            h_t_error = h_t_choice[1]
            h_t_c = j_hypo_c[feature_dim_j]
            h_t_label = j_hypo_label[feature_dim_j]
            h_t = [feature_dim_j, h_t_c, h_t_label]
            self.H.append(h_t)
            # if error is too small, stop
            if h_t_error < self.error_max:
                self.alpha[t] = 1.0
                return

            # calculate alpha
            self.alpha[t] = ((1.0/2.0) * math.log( (1.0 - h_t_error) / h_t_error ))

            if h_t_label is 1:
                h_t_binary_label = 1
                h_t_binary_other_label = -1
            else:
                h_t_binary_label = -1
                h_t_binary_other_label = 1

            # adjust distribution D
            Z_t = 0.0
            for i in range(n):
                x_i = instances[i]
                y_i = self.get_binary_int_label(x_i.get_label())
                x_i_j = x_i.get_feature_vector().get(feature_dim_j+1)
                if x_i_j > h_t_c:
                    h_t_x_i = h_t_binary_label
                else:
                    h_t_x_i = h_t_binary_other_label
                self.D[i] = ( self.D[i] * math.exp(-1.0 * self.alpha[t] * y_i * h_t_x_i) )
                Z_t += self.D[i]

            for i in range(n):
                self.D[i] = self.D[i] / Z_t

    # Prediction
    def predict(self, instance):
        arg_max_one = 0
        arg_max_zero = 0
        for t in range(len(self.H)):
            y_hat = self.hypo_prediction(self.H[t], instance)
            if y_hat is 1:
                arg_max_one += self.alpha[t]
            else:
                arg_max_zero += self.alpha[t]
        if arg_max_one > arg_max_zero:
            return ClassificationLabel(1)
        else:
            return ClassificationLabel(0)

    def get_binary_int_label(self, label):
        if label.get_int_label() is 1:
            return 1
        else:
            return -1

    def hypo_prediction(self, hypothesis, instance):
        j = hypothesis[0]
        c = hypothesis[1]
        label = hypothesis[2]
        x_j = instance.get_feature_vector().get(j+1)
        if (x_j > c) and label is 1:
            return 1
        elif (x_j <= c) and label is 1:
            return 0
        elif (x_j > c) and label is 0:
            return 0
        else:
            return 1