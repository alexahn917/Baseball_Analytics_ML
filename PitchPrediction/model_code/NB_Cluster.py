# Author: SangHyeon (Alex) Ahn
import math, operator
from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor

class NB_Cluster(Predictor):
    def __init__(self, num_clusters, cluster_iterations):
        self.K = num_clusters
        self.K_clusters = [] #k-th [mean, variance, size, probability, cluster num]
        self.max_feature = 0
        self.N = 0
        self.iterations = cluster_iterations

    def train(self, instances):
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

        initial_cluster_elems = []
        for k in range(self.K):
            self.K_clusters.append([FeatureVector(),FeatureVector(), 0, 0, 0]) 
            initial_cluster_elems.append([0, []]) #k-th [size, instances]

        self.N = len(instances)
        for i in range(self.N):
            initial_cluster_elems[i%self.K][0] += 1
            initial_cluster_elems[i%self.K][1].append(instances[i])
        self.updateClusters(initial_cluster_elems, instances, 1)

        for step in range(self.iterations):
#            self.printClusters(step)
            # E-step
            instances_cluster_assignments = []
            for instance in instances:
                instances_cluster_assignments.append([self.argMaxCondProb(self.K_clusters, instance), instance])

            # M-step
            # (1) update probabilities
            new_cluster = []
            for k in range(self.K):
                new_cluster.append([0, []])
            for entry in instances_cluster_assignments:
                i = entry[0]
                new_cluster[i][0] += 1
                new_cluster[i][1].append(entry[1])
            # (2) update means and variances
            self.updateClusters(new_cluster, instances, 0)
        self.printClusters(step)

    def predict(self, instance):        
        return ClassificationLabel(self.argMaxCondProb(self.K_clusters,instance))

    def updateClusters(self, cluster_elems, instances, initial):
        updated_clusters = self.K_clusters
        for k in range(self.K):
            updated_clusters[k][2] = cluster_elems[k][0]
            # empty cluster handling
            if updated_clusters[k][2] is 0:
                updated_clusters[k][0] = FeatureVector()
                updated_clusters[k][1] = FeatureVector()
            else:
                updated_clusters[k][0] = self.sampleMean(cluster_elems[k][0], cluster_elems[k][1])
                updated_clusters[k][1] = self.sampleVar(cluster_elems[k][0], cluster_elems[k][1], updated_clusters[k][0], instances)
            if initial:
                updated_clusters[k][3] = cluster_elems[k][0] / (self.N+0.0)
            else:
                updated_clusters[k][3] = (cluster_elems[k][0] + 1.0) / (self.N + self.K)
            updated_clusters[k][4] = k
        self.K_clusters = updated_clusters

    def argMaxCondProb(self, k_clusters, instance):
        cluster_cond_prob_list = []
        for cluster in k_clusters:
            cluster_cond_prob_list.append([cluster, self.probXgivenY(instance, cluster)])
        cluster_cond_prob_list.sort(key=lambda tup:tup[1], reverse=True)
        best_cluster = cluster_cond_prob_list.pop(0)[0]
        return best_cluster[4]

    def probXgivenY(self, instance, cluster):
        logSumProb = 0
        fv_i = instance.get_feature_vector()
        cluster_mean = cluster[0]
        cluster_var = cluster[1]
        for j in range(1, self.max_feature+1):
            x_ij = fv_i.get(j)
            prob = self.logdnorm(x_ij, cluster_mean.get(j), cluster_var.get(j))
            logSumProb += prob 
        return logSumProb + math.log(cluster[3])

    def logdnorm(self, x, mu, var):
        if var == 0.0:
            return 0.0
        frac = 1 / (math.sqrt(2.0*var*math.pi))
        expon = -((x-mu)**2)/(2.0*var)
        prob = math.log(frac) + expon
        return prob

    def sampleMean(self, cluster_size, cluster_instances):
        feature_avgs = FeatureVector()
        for instance in cluster_instances:
            fv_i = instance.get_feature_vector()
            for index, value in fv_i:
                feature_avgs.add(index, feature_avgs.get(index) + value)
        for index, sum_value in feature_avgs:
            feature_avgs.add(index, sum_value / cluster_size)
        return feature_avgs

    def sampleVar(self, cluster_size, cluster_instances, cluster_mean, instances):
        if cluster_size > 1:
            total_feature_var = {}
            feature_sigmas = FeatureVector()

            for j in range(1, self.max_feature+1):
                total_feature_var.update({j:self.FeaturesVar(instances, j)})
                for instance in cluster_instances:
                    fv_i = instance.get_feature_vector()
                    msq = (cluster_mean.get(j) - fv_i.get(j))**2
                    feature_sigmas.add(j, feature_sigmas.get(j) + msq)

            for index, sum_value in feature_sigmas:
                sample_var = (sum_value / (cluster_size-1.0))
                total_var = total_feature_var.get(index)
                if sample_var > total_var:
                    feature_sigmas.add(index, sample_var)
                else:
                    feature_sigmas.add(index, total_var)
            return feature_sigmas
        else:
            return self.InstancesVar(instances)

    def InstancesVar(self, instances):
        feature_sigmas = FeatureVector()
        for j in range(1, self.max_feature+1):
            feature_sigmas.add(j, self.FeaturesVar(instances, j))
        return feature_sigmas

    def FeaturesVar(self, instances, j):
        N = len(instances)
        # compute average of all j-th feature values
        feature_avg = 0
        for instance in instances:
            fv_i = instance.get_feature_vector()
            feature_avg += fv_i.get(j)
        feature_avg = feature_avg/N
        # compute variance of all j-th feature values
        feature_sigma = 0
        for instance in instances:
            fv_i = instance.get_feature_vector()
            feature_sigma += (fv_i.get(j) - feature_avg)**2
        feature_sigma = feature_sigma/(N-1)
        return 0.01*feature_sigma

    def printClusters(self, iterations):
        print '[At iteration %s ] \n' %iterations
        for k in range(self.K):
            print '--------- Cluster %d ------------' %k
            print 'size: %d' %self.K_clusters[k][2]
            print 'mean: %s' %str(self.K_clusters[k][0])
            print 'variance: %s' %str(self.K_clusters[k][1])
            print 'probability: %.4f' %self.K_clusters[k][3]
            print '\n'