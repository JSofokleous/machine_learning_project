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
    success, fail = 0, 0

    # [Distance, index]
    for neighbor in neighbors:
        row_index = neighbor[1]
        if y_train.iloc[row_index] == 0: 
            fail += 1
        elif y_train.iloc[row_index] == 1:
            success += 1 

    if success > fail: return 1
    else: return 0