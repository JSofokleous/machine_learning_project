import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def make_two(classifier, X_train, y_train):
    ax = plt.gca()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdYlBu', alpha=0.25)

    # Split the range into 30 equal parts and return in a 1D list
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    # Return two 2D lists combining the coordinates in xx and yy
    # [[1, 1, ... * 30], [2, 2, ... * 30], ... [30, 30, ... *30]]
    YY, XX = np.meshgrid(yy, xx)

    # Ravel flattens each array into 1D and Vstack joins the two 1D arrays into a 2D array, which is transposed 
    # [[x1, y1], [x2, y2], ... [xn, yn]]
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = classifier.decision_function(xy)
    Z = Z.reshape(XX.shape) 

    # Show decision boundary
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Show support vectors
    # ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=50, alpha=0.25, linewidth=1, facecolors='none', edgecolors='k')
    plt.show() 

def make_three(classifier, X_train, y_train):
    # 
    r = np.exp(-(X_train ** 2).sum(1))

    # Create 3D graph
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X_train[:,0], X_train[:,1], r, c=y_train, s=50, cmap='RdYlBu')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Age')
    ax.set_zlabel('r')

    # Split the range into 30 equal parts and return in a 1D list
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    zz = np.linspace(zlim[0], zlim[1], 30)

    # Return two 2D lists combining the coordinates in xx and yy
    # [[1, 1, ... * 30], [2, 2, ... * 30], ... [30, 30, ... *30]]
    YY, XX = np.meshgrid(yy, xx)

    # Ravel flattens each array into 1D and Vstack joins the two 1D arrays into a 2D array, which is transposed 
    # [[x1, y1], [x2, y2], ... [xn, yn]]
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = classifier.decision_function(xy)
    Z = Z.reshape(XX.shape) 

    # Show decision boundary
    ax.contour(XX, YY, Z ,colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.show()

