import numpy as np
import matplotlib.pyplot as plt

def l2(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K=K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
    
    def _create_clusters(self):
        clusters = [[] for _ in range(self.K)]
        # assign sample to closest centroid
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample):
        # distance of sample to each centroid
        distances = [l2(sample, point) for point in self.centroids]
        return np.argmin(distances)
    
    def _get_centroids(self):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_feautes))
        for idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids
    
    def _is_converged(self, old_centroids):
        # distance between old and new centroids
        distances = [l2(old_centroids[i], self.centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def _get_cluster_labels(self):
        labels = np.empty(self.n_samples)
        for idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = idx
        return labels
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
    
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_feautes = X.shape
        
        # initialize centroids
        random_indices = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_indices]
        
        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to the closest centroid
            self.clusters = self._create_clusters()
            
            if self.plot_steps:
                self.plot()
            
            # calculate new centroids
            old_centroids = self.centroids
            self.centroids = self._get_centroids()
            
            if self._is_converged(old_centroids):
                break
            
            if self.plot_steps:
                self.plot()
        
        # classify samples
        return self._get_cluster_labels()

if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()
            
            