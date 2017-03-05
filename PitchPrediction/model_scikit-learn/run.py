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
from sklearn.model_selection import cross_val_score

def main():
    with open("../pitchers.txt", "r") as f:
      names = f.read().split('\n')
      print(names)

#    for pitcher_name in names:
#      data = readCSV(pitcher_name)
#      train(data, pitcher_name)
#      predict(data, pitcher_name)
    data = readCSV("Jake Arrieta")
    train(data, "Jake Arrieta")

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

#    clf = OneVsRestClassifier(LinearSVC(random_state=0))
#    clf = OneVsOneClassifier(LinearSVC(random_state=0))
#    clf = svm.SVC(decision_function_shape='ovr')
#    clf = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), param_grid)
#    clf = MLPClassifier()

    clf = RandomForestClassifier()
#    clf.fit(X_train, y_train)
#    eval(clf, y_test, clf.predict(X_test))#

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
    eval(grid_search.best_estimator_, y_test, grid_search.best_estimator_.predict(X_test))
       
    # save the classifier
    file_name = "../classifiers/" + pitcher_name + ".pkl"
    joblib.dump(grid_search.best_estimator_, file_name)

def predict(data, pitcher_name):
    file_name = "../classifiers/" + pitcher_name + ".pkl"
    clf = joblib.load(file_name)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.5)
    predictions = clf.predict(X_test)
    eval(clf, y_test, predictions)
    write_results(y_test, predictions, pitcher_name)

def scores(gs, X_train, y_train)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=2)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

def eval(clf, act, pred):
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(act, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(act, pred))


def write_results(act, pred, pitcher_name):
    with open("results.txt", "a") as f:
      f.write(pitcher_name + ":\n")
      f.write("----------------------------------------------------\n")
      f.write(metrics.classification_report(act,pred))
      f.write("\n\n")

if __name__ == "__main__":
    main()