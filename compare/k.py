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

def distance(data_point, sample_features):
    squared_difference = 0
    # Datapoint: [1, 2, 3, 4]
    # Samplepoint: [[1.3, -1.5, 1.8, -0.5, 4.9]]
    for i in range(len(data_point)):
        squared_difference += (data_point[i] - sample_features[0][i]) ** 2
        final_distance = squared_difference ** 0.5
        return final_distance

def classify(sample_features, X_train, y_train, k):

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, cmap='RdYlBu', alpha=0.25)
    ax.scatter(sample_features[:,0], sample_features[:,1], sample_features[:,2], c='k', marker='o', s=150)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Age')
    ax.set_zlabel('Class')

    distances = []
    # Looping through all points in the dataset X_train
    for row_index in range(len(X_train)):
        data_point = X_train[row_index]
        distance_to_point = distance(data_point, sample_features)

        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, row_index])

    # Taking only the k closest points
    distances.sort()
    neighbors = distances[0:k]
    
    # Determine labels of each neighbour
    success, fail = 0, 0
    for neighbor in neighbors:
        row_index = neighbor[1]
        # Add neighbors to scatter
        ax.scatter(X_train[row_index][0], X_train[row_index][1], X_train[row_index][2], c='dimgrey', marker='1', s=300)

        if y_train.iloc[row_index] == 0: 
            fail += 1
        elif y_train.iloc[row_index] == 1:
            success += 1 

    plt.show()

    # Classify point based on majority of neighbours
    if success > fail: return 1
    elif fail > success: return 0

    # If equal, return label of 
    else: 
        print('Equal number of neighbours!')
        return y_train.iloc(neighbors[0][1])
