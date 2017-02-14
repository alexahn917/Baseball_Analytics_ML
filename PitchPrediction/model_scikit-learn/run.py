import csv
import itertools
import warnings
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# ignore convergence warning
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def readCSV():
    csv_file = '../ETL pipeline/raw_data/Clayton Kershaw.csv'

    file = open(csv_file, "r")
    reader = csv.reader(file)

    instances = []
    target = []

    row_num = 0 
    for row in reader:
        if row_num is 0:
            header = row
        else:
            col_num = 0
            features = []
            for col in row:
                if col_num is 0:
                    target.append(int(col))
                    instances.append([])
                else:
                    instances[row_num-1].append(int(col))
                col_num += 1
        row_num +=1
    file.close()

    data = [instances, target]
    return data

def main():
    data = readCSV()
    X = data[0]
    Y = data[1]
    n_samples = len(X)

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#    classifier = OneVsRestClassifier(LinearSVC(random_state=0))
#    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
#    classifier = svm.SVC(decision_function_shape='ovr')
#    classifier = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), param_grid)
#    classifier = MLPClassifier()

    # make grid search for parameters
#    parameters = {
#        'kernel': ['rbf'],
#        'C': [10],
#        'gamma': [0.001]
#    }

#    classifier = GridSearchCV(svm.SVC(decision_function_shape='ovr', verbose=False), parameters)

#    parameters = {
#        'hidden_layer_sizes': [(1000,)],
#        'activation': ['relu'],
#        'alpha': [0.1, 0.01, 10],
#        'algorithm': ['sgd'],
#        'tol': [0.01],
#        'learning_rate': ['invscaling']
#    }
#    classifier = GridSearchCV(MLPClassifier(verbose=True), parameters)


    # Neural Network Classifier
    nn = MLPClassifier()
    nn.fit(X_train, y_train)
    points = nn.score(X_test, y_test)
    print("Neural Network:", points)

    y_pred = nn.predict(X_test)
    
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))


    # Hyper Parameter Tuning using GridSearch (optimization)
    print("\nPerforming parameter tuning for Neural Network Regression model:")
    layers = list(itertools.product(range(1,15), repeat = 2))
    alphas = [0.01, 0.01, 0.01, 0.1, 1.0, 10.0]
    param_grid = {"hidden_layer_sizes": layers, "alpha": alphas}
    NN = GridSearchCV(estimator = MLPClassifier(), param_grid = param_grid,
    scoring = "accuracy", cv = 2, n_jobs = -1)
    NN.fit(X_train, y_train)

    print("Best Estimator:\n", NN.best_estimator_)
    print("Best neg.mean.squared.error Score:\n", NN.best_score_)

    y_pred = NN.best_estimator_.predict(X_test)
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()