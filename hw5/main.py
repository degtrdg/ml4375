import sys
import numpy as np
import pandas as pd

def N(x, mean, variance):
  return (1/np.sqrt(variance*2*np.pi))*np.exp(-0.5*(x - mean)**2/variance)

def prob_x_c(x, phi, mean, variance):
  return phi*N(x,mean,variance)

# define distributions as list of tuples = [(phi probability of belonging to cluster 1, mean of cluster 1, variance of cluster 1), (phi 2, mean 2, variance 2), ...]
def gamma(x, distributions):
  probs = [prob_x_c(x, *i) for i in distributions]
  return list(probs/sum(probs))

def calc_gammas(values, distributions):
  return [gamma(x, distributions) for x in values]

# define gammas as list of lists = [[gamma for point 1 cluster 1, gamma for point 1 cluster 2, ..], [gamma for point 2 cluster 1, gamma for point 2 cluster 2, ..] ...]
# gammas_for_each_cluster = [[gamma for point 1 cluster 1, gamma for point 2 cluster 1, ..], [gamma for point 1 cluster 2, gamma for point 2 cluster 2, ..] ...]

def new_cluster_phis(gammas):
  gammas_for_each_cluster = np.array(gammas).T.tolist()
  return [np.mean(cluster) for cluster in gammas_for_each_cluster]

def new_cluster_means(values, gammas):
  gammas_for_each_cluster = np.array(gammas).T.tolist()
  sums_of_clusters = [sum(cluster) for cluster in gammas_for_each_cluster]
  weighted_sums = [np.dot(values, cluster) for cluster in gammas_for_each_cluster]
  return [i[0]/i[1] for i in list(zip(weighted_sums, sums_of_clusters))]

def weighted_sq_diff(gamma, x, mean):
  return gamma*(x - mean)**2

def sum_weighted_sq_diff(gammas, xs, means):
  params_for_each_point = list(zip(gammas, xs, means))
  return sum([weighted_sq_diff(*point) for point in params_for_each_point])

def new_cluster_variances(values, means, gammas):
  num_points = len(values)
  num_means = len(means)
  values = [values for i in range(num_means)]
  means = [[m]*num_points for m in means]
  gammas_for_each_cluster = np.array(gammas).T.tolist()
  sums_of_clusters = [sum(cluster) for cluster in gammas_for_each_cluster]

  numerators = [sum_weighted_sq_diff(*cluster) for cluster in list(zip(gammas_for_each_cluster, values, means))]
  return [i[0]/i[1] for i in list(zip(numerators, sums_of_clusters))]

def one_hot(index, length):
  result = [0]*length
  result[index] = 1
  return result








def main(train_file, num_clusters, num_iterations):
    num_clusters = int(num_clusters)
    num_iterations = int(num_iterations)

    data_tbl = pd.read_csv(train_file, header=None)
    data = list(data_tbl[0])
    gammas = [one_hot(i%num_clusters, num_clusters) for i,_ in enumerate(data)]

    for i in range(num_iterations+1):
        means = new_cluster_means(data, gammas)
        variances = new_cluster_variances(data, means, gammas)
        priors = new_cluster_phis(gammas)
        distributions = list(zip(priors, means, variances))
        gammas = calc_gammas(data, distributions)

        print(f"After iteration {i}:")
        for j,dist in enumerate(distributions):
            print(f"Gaussian {j+1}: mean = {dist[1]:.4f}, variance = {dist[2]:.4f}, prior = {dist[0]:.4f}")
        print()


if __name__ == "__main__":
    train_file = sys.argv[1]
    num_clusters = sys.argv[2]
    num_iterations = sys.argv[3]
    main(train_file, num_clusters, num_iterations)

    
    