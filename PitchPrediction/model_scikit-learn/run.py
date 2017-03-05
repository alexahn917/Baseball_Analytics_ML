import csv
import pickle
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.externals import joblib

def main():
    pitcher_name = "Clayton Kershaw"
    data = readCSV(pitcher_name)
    train(data, pitcher_name)

def readCSV(pitcher_name):
    csv_file = '../ETL pipeline/csv_data/' + pitcher_name + '.csv'
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

def train(data, pitcher_name):
    X = data[0]
    y = data[1]
    n_samples = len(X)

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#    classifier = OneVsRestClassifier(LinearSVC(random_state=0))
#    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
#    classifier = svm.SVC(decision_function_shape='ovr')
#    classifier = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), param_grid)
#    classifier = MLPClassifier()

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    eval(clf, y_test, clf.predict(X_test))
#    file_name = "../classifiers/" + pitcher_name + ".pkl"
#    joblib.dump(clf, file_name)


    # run grid search
    param_grid = {'n_estimators': [10, 500],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  "max_depth": [3, 5, None],
                  "min_samples_leaf": [1, 3, 5],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_jobs": [-1]
                  }

    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    eval(grid_search.cv_results_, y_test, grid_search.best_estimator_.predict(X_test))
    # save the classifier
    file_name = "../classifiers/" + pitcher_name + ".pkl"
    joblib.dump(grid_search.best_estimator_, file_name)

    
'''
    # run randomized search
    param_dist = {'n_estimators': [100, 1000],
                  "max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]
                  }


    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)
    random_search.fit(X_train, y_train)
    eval(random_search.cv_results_, y_test, random_search.best_estimator_.predict(X_test))
'''

def eval(result, act, pred):
    print("Classification report for classifier %s:\n%s\n"
      % (result, metrics.classification_report(act, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(act, pred))
    
if __name__ == "__main__":
    main()