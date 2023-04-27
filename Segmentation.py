import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import feature
from sklearn.cluster import KMeans


def main():
    D = io.imread("data/attention-mri.tif")
    print(D.shape)

    im_x = D[63, :, :]  # sagittal(x)
    im_y = D[:, 63, :]  # coronal(y)
    im_z = D[:, :, 15]  # transaxial(z)

    plt.subplot(1, 3, 1), plt.imshow(im_x, cmap='gray', aspect=0.5)
    plt.title('Sagittal'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(im_y, cmap='gray', aspect=0.5)
    plt.title('Coronal'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(im_z, cmap='gray')
    plt.title('Axial'), plt.xticks([]), plt.yticks([])
    plt.show()

# def Edge_Filters(): (1):
    Gx = cv2.Sobel(im_z, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(im_z, cv2.CV_64F, 0, 1, ksize=3)
    Gmat = np.sqrt(Gx**2.0 + Gy**2.0)

    plt.subplot(1, 3, 1), plt.imshow(Gx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(Gy, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(Gmat, cmap='gray')
    plt.title('Final'), plt.xticks([]), plt.yticks([])
    plt.show()

    # (2):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Gx = cv2.filter2D(im_z, -1, kernel_x)
    Gy = cv2.filter2D(im_z, -1, kernel_y)
    Gmat = np.sqrt(Gx**2.0 + Gy**2.0)

    plt.subplot(1, 3, 1), plt.imshow(Gx, cmap='gray')
    plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(Gy, cmap='gray')
    plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(Gmat, cmap='gray')
    plt.title('Final'), plt.xticks([]), plt.yticks([])
    plt.show()

    # (3):
    ef1 = feature.canny(im_z, 1.0, 2, 5)
    ef2 = feature.canny(im_z, 1.0, 3, 15)

    plt.subplot(1, 2, 1), plt.imshow(ef1, cmap='gray')
    plt.title('(2, 5)'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(ef2, cmap='gray')
    plt.title('(3, 15)'), plt.xticks([]), plt.yticks([])
    plt.show()

# def Kmeans_clustering(): (2):
    count = 1
    X = im_z.reshape((-1, 1))

    estimators = [
        ('kmeans_4', KMeans(n_clusters=4)),
        ('kmeans_8', KMeans(n_clusters=8)),
        ('kmeans_20', KMeans(n_clusters=20))
    ]

    for name, est in estimators:
        kmeans = est.fit(X)
        labels = kmeans.labels_
        choices = kmeans.cluster_centers_.squeeze()
        img = np.choose(labels, choices)
        img.shape = im_z.shape
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 3, count), plt.imshow(img, plt.cm.Spectral)
        plt.title(name)
        count += 1
        plt.show()

    # (3):
    trytime = 500
    x_list = []
    y_list = []

    for i in range(trytime):
        kmeans = KMeans(n_clusters=4).fit(im_z)
        x_list.append(kmeans.n_iter_)
        y_list.append(kmeans.inertia_)

    ax = plt.gca()
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Within-cluster sums')
    ax.scatter(x_list, y_list, c='r', s=20, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
