from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd

## CLASSIFIER
def fit_score_model(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)

    # Score the model 
    model.score(X_train, y_train)
    model.score(X_test, y_test)

    # Analyse coefficients by printing:
    #### AttributeError: coef_ is only available when using a linear kernel
    # list(zip(['Sex','Age','FirstClass','SecondClass', 'Master'],model.coef_[0]))

    # Predict labels using test data
    y_pred = model.predict(X_test)

    # Determine accuracy and F1 score, Round to 1.d.p and convert to percentage 
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(100*accuracy, 1)
    f1 = f1_score(y_test, y_pred)
    f1 = round(100*f1, 1)

    return accuracy, f1



## KNN MODEL
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def get_best_k(X_train, y_train, X_test, y_test):
    k_list = range(1, 101)
    scores = []
    best_score, best_k = 0, 0
    for k in k_list:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)
        if score > best_score: 
            best_score = score
            best_k = k
    # plt.plot(k_list, scores)
    # plt.show()
    return best_k

def k_distance(data_point, sample_features, feature_names_list):
    squared_difference = 0
    # Datapoint: [1, 2, 3, 4]
    # Samplepoint: [[1.3, -1.5, 1.8, -0.5, 4.9]]
    for i in range(len(data_point)):
        squared_difference += (data_point[feature_names_list[i]].item() - sample_features[i]) ** 2
        final_distance = squared_difference ** 0.5
        return final_distance

def k_classify(sample_features_norm, X_train_norm, y_train, k, feature_names_list, sample_features, X_train):

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[feature_names_list[0]], X_train[feature_names_list[1]], X_train[feature_names_list[2]], c=y_train, cmap='RdYlBu', alpha=0.15)
    ax.scatter(sample_features[0], sample_features[1], sample_features[2], c='k', marker='o', s=300)
    ax.set_xlabel(feature_names_list[0])
    ax.set_ylabel(feature_names_list[1])
    ax.set_zlabel(feature_names_list[2])

    ## DETERMINE AND PLOT CLOSEST NEIGHBOURS
    # Loop through all points in the dataset X_train
    distances = []
    for row_index in range(len(X_train)):
        data_point = X_train.loc[[row_index]]
        distance_to_point = k_distance(data_point, sample_features, feature_names_list)
        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, row_index])

    # Taking only the k closest points
    distances.sort()
    neighbors = distances[0:k]

    # Classify point based on majority of neighbours (If equal, return label of FIRST neighbour)
    success, fail = 0, 0
    for neighbor in neighbors:
        row_index = neighbor[1]
        # Add neighbors to scatter
        row = X_train.loc[[row_index]]
        ax.scatter(row[feature_names_list[0]].item(), row[feature_names_list[1]].item(), row[feature_names_list[2]].item(), c='dimgrey', marker='1', s=500)

        if y_train.iloc[row_index] == 0: 
            fail += 1
        elif y_train.iloc[row_index] == 1:
            success += 1 


    plt.show()
       
    if success > fail: return 1
    elif fail > success: return 0
    else: 
        print('Equal number of neighbours!')
        return y_train.iloc[neighbors[0][1]]


## SVM MODEL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def two_d_svm(classifier, X_train, y_train):
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

def three_d_svm(classifier, X_train, y_train):
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

