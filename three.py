import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

def generate1000():
    randomNumber = np.random.normal()    
    print(f'Random number {randomNumber}')

    randomNumbers = np.random.normal(size=10)
    print(f'Random numbera {randomNumbers}')
    mean = [0,0]
    covariance = [[64,36], [36,30]]
    x, y = np.random.multivariate_normal(mean, covariance, 1000).T
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()

    angle = np.radians(30.)
    x1 = np.random.normal(scale=8, size=1000)
    x2 = x1*np.tan(angle) + np.random.normal(scale=3, size=1000)
    x1 = x1-np.mean(x1)
    x2 = x2-np.mean(x2)
    data = np.array([x1,x2])
    # plt.plot(x1, x2, 'x')
    # plt.axis('equal')
    # plt.show()

    pca = PCA(n_components=1)
    pca.fit(data.T)
    first_component = pca.transform(data.T)
    # first_component_xy = pca.inverse_transform(first_component)

    
    digits = load_digits()
    print(digits.data.shape)
    # plt.gray() 
    # plt.matshow(digits.images[6]) 
    # plt.show()   

    pca = PCA(50)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(digits.data)
    print(digits.data.shape)
    print(projected.shape)

    plt.scatter(projected[:, 0], projected[:, 1],
            c= digits.target, edgecolor='none', alpha=0.5,
            cmap= plt.cm.get_cmap('pink', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()



if __name__ == "__main__":