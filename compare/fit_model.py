from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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