# coding: utf-8
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
    for pitcher_name in names:
      train_data = readCSV(pitcher_name, 'train')
      test_data = readCSV(pitcher_name, 'test')      
      train(train_data, pitcher_name)
      predict(test_data, pitcher_name)


def readCSV(pitcher_name, file_type):
    csv_file = '../ETL pipeline/CSV/extended/' + pitcher_name + '_' + file_type +'.csv'
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

#    clf = OneVsRestClassifier(LinearSVC(random_state=0))
#    clf = OneVsOneClassifier(LinearSVC(random_state=0))
#    clf = svm.SVC(decision_function_shape='ovr')
#    clf = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), param_grid)
    clf = MLPClassifier()
#    clf = RandomForestClassifier()

    clf.fit(X, y)
  
    # grid search
#    param_grid = get_param_grid('MLP')
#    grid_search = GridSearchCV(clf, param_grid=param_grid)
#    grid_search.fit(X, y)
#    clf = grid_search.best_estimator_
       
    # save the classifier
    file_name = "../classifiers/extended/" + pitcher_name + ".pkl"
    joblib.dump(clf, file_name)


def predict(data, pitcher_name):
    X = data[0]
    y = data[1]
    file_name = "../classifiers/extended/" + pitcher_name + ".pkl"
    clf = joblib.load(file_name)
    predictions = clf.predict(X)
    eval(clf, y, predictions)
    write_results(y, predictions, pitcher_name)


def get_param_grid(model):
    if (model == 'MLP'):
      param_grid = {'hidden_layer_sizes': [(10, ), (25, ), (50, ), (75, ), (100, )],
                    'activation' : ['relu', 'logistic', 'tanh', 'identity'],
                    'solver' : ['lbfgs', 'sgd', 'adam'],
                    'alpha' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                    'batch_size' : [200,400,600,800,1000],                    
                    }
    if (model == 'RandomForest'):
      param_grid = {'n_estimators': [10, 500],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    "max_depth": [3, 5, None],
                    "min_samples_leaf": [1, 3, 5],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    "n_jobs": [-1]
                    }
    return param_grid


def scores(gs, X_train, y_train):
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=2)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def eval(clf, act, pred):
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(act, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(act, pred))


def write_results(act, pred, pitcher_name):
    with open("../classifiers/extended/nn-results.txt", "a") as f:
      f.write(pitcher_name + ":\n")
      f.write("----------------------------------------------------\n")
      f.write(metrics.classification_report(act,pred))
      f.write("\n\n")


if __name__ == "__main__":
    main()