import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error

def pca_and_reconstruct(num_components, images, labels):
    pca = PCA(num_components)
    pca_project = pca.fit_transform(images)
    print(f'digits {images.shape}')
    print(f'PCA {pca_project.shape}')

    plt.scatter(pca_project[:,0], pca_project[:,1],
         c=labels, alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.colorbar() 
    plt.show() 
        
    images_reconstructed = pca.inverse_transform(pca_project)
    print(f'Reconstructed {images_reconstructed.shape}')

    last_image = images[-1]
    last_image_label = labels[-1]
    last_image_reconstructed = images_reconstructed[-1]

    mse_last_image = (np.square(last_image - last_image_reconstructed)).mean(axis=None)    
    print(f'Mean Squared Error for last digit - {last_image_label} with {num_components} principal components is {mse_last_image}')
    plt.matshow(last_image.reshape(16,16))
    plt.matshow(last_image_reconstructed.reshape(16,16))
    plt.show()

    second_last_image = images[-2]
    second_last_image_label = labels[-2]
    second_last_image_reconstructed = images_reconstructed[-2]

    mse_second_last_image = (np.square(second_last_image - second_last_image_reconstructed)).mean(axis=None)    
    print(f'Mean Squared Error for second last digit - {second_last_image_label} with {num_components} principal components is {mse_second_last_image}')
    plt.matshow(second_last_image.reshape(16,16))
    plt.matshow(second_last_image_reconstructed.reshape(16,16))
    plt.show()

def load_usps():
    usps_data = loadmat('C:\\repos\\qna\\src\\atasks\\USPS')
    print(f'USPS data keys - {usps_data.keys()}')
    return usps_data

def four():
    # Loads usps data set from usps.mat
    usps = load_usps()
    # X contains the data for all images 16X16
    data = usps['X']
    
    # loads the labels for each image array.
    labels = [t[0]-1 for t in usps['Y']]
    
    # pca with p=10,50,100,150,200 components for the entire data set
    pca_and_reconstruct(10, data, labels)
    pca_and_reconstruct(50, data, labels)
    pca_and_reconstruct(100, data, labels)
    pca_and_reconstruct(150, data, labels)
    pca_and_reconstruct(200, data, labels)

    # thoughts on task
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('variance')
    plt.show()

def three():
    # 3a: 1000 data points generated from a normal distribution
    # Mean (top point of the bell curve) is at 0,0
    # Covariance chosen to get points distributed around a 30 degree angle
    print(f'a. Generating 1000 data points in 2 dimensions')
    mean = [0,0]
    covariance = [[64,36], [36,30]]
    x, y = np.random.multivariate_normal(mean, covariance, 1000).T
    data = np.array([x, y]).T
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

    # 3b
    pca = PCA(n_components=1)
    pca.fit(data)
    print("PCA component calculated:", pca.components_)    
    print("PCA explained variance:", pca.explained_variance_)

    first_component = pca.transform(data)

    plt.scatter(first_component[:, 0], np.zeros(1000) , alpha=0.2)
    plt.axis('equal')
    plt.show()

    # 3c
    first_component_xy = pca.inverse_transform(first_component)    
    print("reconstructed shape:", first_component_xy.shape)

    plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
    plt.scatter(first_component_xy[:, 0], first_component_xy[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    three()
    four()