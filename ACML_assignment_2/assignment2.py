import numpy as np
import matplotlib.pyplot as plt

class kMeans:
    def __init__(self, k, dataset, initalCentres):
        self.k = k
        self.dataset = dataset
        self.clusters = []
        for i in range(k):
            cluster = {
                "mean": initalCentres[i],
                "points": []
            }
            self.clusters.append(cluster)

    def train(self):
        converged = False
        while not converged:
            for c in self.clusters:
                c["points"] = []

            for d in self.dataset:
                norm = [np.linalg.norm(np.array(d) - np.array(c["mean"])) for c in self.clusters]
                min_index = norm.index(min(norm))
                self.clusters[min_index]["points"].append(d)

            converged = True 

            for c in self.clusters: 
                if c["points"]:
                    new_mean = np.mean(c["points"], axis=0)
                    if not np.allclose(c["mean"], new_mean):
                        converged = False  # means changed => not converged
                    c["mean"] = new_mean

        error = 0.0
        for c in self.clusters:
            for p in c["points"]:
                error += np.linalg.norm(np.array(p) - np.array(c["mean"]))**2
        return error
    
    def plot(self):
        for c in self.clusters:
            points = np.array(c["points"])
            plt.scatter(points[:, 0], points[:, 1])
            plt.scatter(c["mean"][0], c["mean"][1], marker="x", color="red", s=100)
        plt.show()
                

if __name__== "__main__":
    x = [0.22, 0.45, 0.73, 0.25, 0.51, 0.69, 0.41, 0.15, 0.81, 0.50, 0.23, 0.77, 0.56, 0.11, 0.81, 0.59, 0.10, 0.55, 0.75, 0.44]
    y = [0.33, 0.76, 0.39, 0.35, 0.69, 0.42, 0.49, 0.29, 0.32, 0.88, 0.31, 0.30, 0.75, 0.38, 0.33, 0.77, 0.89, 0.09, 0.35, 0.55]

    initialCentres = [np.array([float(input()),float(input())]) for _ in range(3)]

    kmeans = kMeans(3, np.array(list(zip(x, y))), initialCentres)
    error = kmeans.train()
    # kmeans.plot()
    print(round(error, 4))