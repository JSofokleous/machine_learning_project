import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def make_two(classifier, X_train, y_train):
    ax = plt.gca()
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=50, cmap='autumn')
    plt.scatter(classifier.support_vectors_[:,0],classifier.support_vectors_[:,1])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
        linestyles=['--', '-', '--'])

    ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100,
        linewidth=1, facecolors='none', edgecolors='k')
    plt.show() 

def make_three(classifier, X_train, y_train):
    r = np.exp(-(X_train ** 2).sum(1))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X_train.Age, X_train.Sex, r, c=y_train, s=50, cmap='autumn')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()


    # fig = plt.figure()
    # ax  = fig.add_subplot(111, projection='3d')
    # # ax.scatter(df.Age, df.Pclass, df.Sex, c=df.Survived, cmap=plt.cm.RdYlBu, alpha=0.5)

