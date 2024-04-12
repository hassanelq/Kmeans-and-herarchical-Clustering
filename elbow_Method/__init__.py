import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class ElbowMethod:
    def __init__(self, data, k_range=(1, 10)):
        self.data = data
        self.k_range = range(*k_range)
        self.distortions = []
        self.optimal_k = None

    def fit(self):
        """ Fit the KMeans model using a range of k and calculate distortions. """
        for k in self.k_range:
            model = KMeans(n_clusters=k)
            model.fit(self.data)
            self.distortions.append(sum(np.min(cdist(self.data, model.cluster_centers_, 'euclidean'), axis=1)) / self.data.shape[0])

    def find_optimal_k(self):
        """ Determine the optimal number of clusters using the Elbow method. """
        # Calculate the difference line
        steep = (self.distortions[-1] - self.distortions[0]) / (len(self.distortions) - 1)
        c = self.distortions[-1] - steep * len(self.distortions)
        linear = [steep * (x + 1) + c for x in range(len(self.distortions))]

        # Calculate distances from actual distortions to the line
        distances = np.abs(np.array(linear) - np.array(self.distortions))
        self.optimal_k = distances.argmax() + 1  # +1 because index starts at 0
        return self.optimal_k

    def plot_elbow_curve(self):
        """ Plot the elbow curve along with the line showing the optimal k determination. """
        fig, ax = plt.subplots()
        ax.plot(self.k_range, self.distortions, 'bx-', label='Distortions', color='blue')
        if self.optimal_k:
            ax.axvline(x=self.optimal_k, linestyle='--', color='red', label=f'Optimal k = {self.optimal_k}')
        ax.set_xlabel('k')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method showing the optimal k')
        ax.legend()
        return fig