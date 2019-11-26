import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/1.csv')
x = df.values


def separate_points_by_clusters(x, centroids, k):
    points_by_clusters = [[] for i in range(k)]
    for i in range(x.shape[0]):
        difference = np.power(x[i] - centroids, 2)
        distances = np.sqrt(np.sum(difference, axis=1))
        index = np.argmin(distances)
        points_by_clusters[index].append(i)
    return points_by_clusters


def get_new_centroids(x, points_by_clusters, old_centroids):
    new_centroids = []
    for i in range(len(points_by_clusters)):
        sum_by_features = np.zeros((15,))
        m = len(points_by_clusters[i])
        if m != 0:
            for j in range(m):
                sum_by_features = sum_by_features + x[points_by_clusters[i][j]]
            centroid = sum_by_features / m
        else:
            centroid = old_centroids[i]
        new_centroids.append(centroid)
    return np.array(new_centroids)


def get_min_distance_to_others_centroids(centroids, point, index):
    min_distance = 1000000000
    for i in range(centroids.shape[0]):
        if i != index:
            difference = np.power(point - centroids[i], 2)
            distance = np.sqrt(np.sum(difference))
            if min_distance > distance:
                min_distance = distance
    return min_distance


def distance_for_own_centroid(centroid, point):
    difference = np.power(point - centroid, 2)
    distance = np.sqrt(np.sum(difference))
    return distance


def silhouette_score(x, centroids, points_by_clusters):
    silhouette_score = 0
    for i in range(len(points_by_clusters)):
        m = len(points_by_clusters[i])
        for j in range(m):
            point = x[points_by_clusters[i][j]]
            b = get_min_distance_to_others_centroids(centroids, point, i)
            a = distance_for_own_centroid(centroids[i], point)
            if max(b, a) != 0:
                silhouette_score += (b - a) / max(b, a)
    return silhouette_score / x.shape[0]


def kmeans(x, k):
    centroid_indexes = np.random.randint(x.shape[0], size=k)
    centroids = x[centroid_indexes, :]
    points_by_clusters = separate_points_by_clusters(x, centroids, k)
    for _ in range(6):
        centroids = get_new_centroids(x, points_by_clusters, centroids)
        points_by_clusters = separate_points_by_clusters(x, centroids, k)
    s = silhouette_score(x, centroids, points_by_clusters)
    return centroids, points_by_clusters, s


start_k = 2
end_k = 150

silhouette_scores = []
k_list = []
for k in range(start_k, end_k):
    s_min = 10
    res_better = []
    for i in range(1):
        res = kmeans(x, k)
        if s_min > res[2]:
            s_min = res[2]
            res_better = res
    print(k)
    silhouette_res = s_min
    k_list.append(k)
    silhouette_scores.append(silhouette_res)

plt.plot(k_list, silhouette_scores)
plt.show()
print('Optimal k: ' + str(np.argmax(np.array(silhouette_scores))))
