# Author: SangHyeon (Alex) Ahn
import math, operator
from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

class LambdaMeans(Predictor):
    def __init__(self, cluster_lambda, cluster_iterations):
        self.Lambda = cluster_lambda
        self.iterations = cluster_iterations
        self.K_cluster_means = []
        self.K = 0
        self.max_feature = 0

    def train(self, instances):
        N = len(instances)
        instance_clusters = [0]*N
        j_curr = 0
        j_max = 0
        for instance in instances:
            try:
                j_curr = max(instance.get_feature_vector().getKeys())
                if j_max < j_curr:
                    j_max = j_curr
            except:
                print('no elements')
        self.max_feature = j_max

        # Initialization
        mu_1 = FeatureVector()
        # Question: aveage of the features that showed up? or still divide it by 1/N?
        for instance in instances:
            feature_vector_i = instance.get_feature_vector()
            for index, value in feature_vector_i:
                mu_1.add(index, mu_1.get(index) + value)
        for index, feature_sum in mu_1:
            mu_1.add(index, feature_sum / (N))
        self.K_cluster_means.append([self.K, mu_1])

        if self.Lambda > 0:
            pass
        else:
            dist = []
            for instance in instances:
                dist.append(self.distance(mu_1, instance)**2)
            self.Lambda = sum(dist)/(N)

        # Do E-M computation for clusters
        for i in range(self.iterations):            
            instances_cluster_assignments = []
            clusters = self.K_cluster_means
            # E: assign cluster to each instance
            for instance in instances:
                dist_to_cluster_k = []
                # calculate distance from an instance to each cluster mean
                for cluster in clusters:
                    dist_to_cluster_k.append([cluster[0], self.distance(cluster[1], instance)**2])
                # retrieve closest cluster
                dist_to_cluster_k.sort(key=lambda tup:tup[1])
                min_dist_cluster = dist_to_cluster_k.pop(0)
                # assign cluster if distance is less than or equal to lambda
                if min_dist_cluster[1] <= self.Lambda:
                    instances_cluster_assignments.append([min_dist_cluster[0], instance])
                # otherwise, make a new cluster and K = K+1
                else:
                    self.K = self.K+1
                    clusters.append([self.K, instance.get_feature_vector()])
                    instances_cluster_assignments.append([self.K, instance])
            self.K_cluster_means = clusters

            # Printing clustering results
#            for c in self.K_cluster_means:
#                print('cluster: %d with mean %s' %(c[0], str(c[1])))
#            for i in instances_cluster_assignments:
#                print('instance cluster assignment: instance %s to cluster %d' %(str(i[1].get_feature_vector()), i[0]))

            # M: Update u_k
            clusters = self.K_cluster_means
            for cluster in clusters:
                updated_k_mean = FeatureVector()
                count = 0.0
                for instances_cluster_assignment in instances_cluster_assignments:
                    if instances_cluster_assignment[0] is cluster[0]:
                        for index, value in instances_cluster_assignment[1].get_feature_vector():
                            updated_k_mean.add(index, updated_k_mean.get(index) + value)
                        count = count + 1.0
                # Handle empty clusters
                if count is 0.0:
                    cluster_k_mean = FeatureVector()
                else:
                    # Divide the sum by number of counts
                    for index, value in updated_k_mean:
                        updated_k_mean.add(index, value/count)
                    cluster[1] = updated_k_mean
            self.K_cluster_means = clusters

#        print('Lambda: %f' %self.Lambda)
#        print('Number of Clusters + 1: %d' %self.K)
#        for c in self.K_cluster_means:
#            print('cluster: %d with mean %s' %(c[0], str(c[1])))

    def predict(self, instance):
        clusters = self.K_cluster_means
        dist_to_cluster_k = []
        # calculate distance from an instance to each cluster mean
        for cluster in clusters:
            dist_to_cluster_k.append([cluster[0], self.distance(cluster[1], instance)**2])
        dist_to_cluster_k.sort(key=lambda tup:tup[1])
        min_dist_cluster = dist_to_cluster_k.pop(0)
        return ClassificationLabel(min_dist_cluster[0])

    def distance(self, k_mean, instance_i):
        feature_vector_i = instance_i.get_feature_vector()
        dist_vector = []
        for j in range(self.max_feature + 1):
            dist_vector.append((k_mean.get(j) - feature_vector_i.get(j))**2)
        return math.sqrt(sum(dist_vector))