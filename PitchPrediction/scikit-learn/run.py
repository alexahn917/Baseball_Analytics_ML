# coding: utf-8
import csv
import pickle
import numpy as np
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

    model_names = ['svm', 'nn', 'rf']
    
    clear_txt_files(model_names)
    
    # Train, Test, Write Results
    for pitcher_name in names:
        train_data = readCSV(pitcher_name, 'train')
        test_data = readCSV(pitcher_name, 'test')      
        scores = []
        for model_name in model_names:
            train(train_data, pitcher_name, model_name, True)
            clf = load(pitcher_name)
            scores.append((model_name, predict(clf, test_data, pitcher_name, model_name, True), clf))

        scores.sort(key=lambda tup:tup[1], reverse=True)
        print(scores)
        write_best_scores(scores, pitcher_name)

    # Making predictions
    
    #predict_atbat(clf)

def predict_atbat(clf):
    X = [0] * 23
    val = raw_input("pitch_type: ")
    X[0] = int(val) if val else 0
    val = raw_input("batter_num: ")
    X[1] = int(val) if val else 0
    val = raw_input("pitch_rl: ")
    X[2] = int(val) if val else 0
    val = raw_input("bat_rl: ")
    X[3] = int(val) if val else 0
    val = raw_input("inning: ")
    X[4] = int(val) if val else 0
    val = raw_input("balls: ")
    X[5] = int(val) if val else 0
    val = raw_input("strikes: ")
    X[6] = int(val) if val else 0
    val = raw_input("out: ")
    X[7] = int(val) if val else 0
    val = raw_input("on_1b: ")
    X[8] = int(val) if val else 0
    val = raw_input("on_2b: ")
    X[9] = int(val) if val else 0
    val = raw_input("on_3b: ")
    X[10] = int(val) if val else 0
    val = raw_input("score_diff: ")
    X[11] = int(val) if val else 0
    val = raw_input("era: ")
    X[12] = float(val) if val else 0.0
    val = raw_input("rbi: ")
    X[13] = float(val) if val else 0.0
    val = raw_input("avg: ")
    X[14] = float(val) if val else 0.0
    val = raw_input("hr: ")
    X[15] = int(val) if val else 0
    val = raw_input("pitcher_at_home: ")
    X[16] = int(val) if val else 0
    val = raw_input("pitcher_wins: ")
    X[17] = int(val) if val else 0
    val = raw_input("pitcher_losses: ")
    X[18] = int(val) if val else 0
    val = raw_input("batter_wins: ")
    X[19] = int(val) if val else 0
    val = raw_input("batter_losses: ")
    X[20] = int(val) if val else 0
    val = raw_input("prev_pitch_type: ")
    X[21] = int(val) if val else -1
    val = raw_input("prevprev_pitch_type: ")
    X[22] = int(val) if val else -1
    print(clf.predict(X))


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
                    instances[row_num-1].append(float(col))
                col_num += 1
        row_num +=1
    file.close()
    data = [instances, target]
    return data


def train(data, pitcher_name, model_name, grid_search):
    X = data[0]
    y = data[1]
    n_samples = len(X)
#    clf = OneVsRestClassifier(LinearSVC(random_state=0))
#    clf = OneVsOneClassifier(LinearSVC(random_state=0))
    if model_name == 'svm':
        clf = svm.SVC(decision_function_shape='ovr', gamma='auto', kernel='rbf')
    #    clf = GridSearchCV(svm.SVC(kernel='rbf', decision_function_shape='ovr'), param_grid)
    elif model_name == 'nn':
        clf = MLPClassifier()
    elif model_name == 'rf':
        clf = RandomForestClassifier()
    
    # grid search
    if grid_search:
        param_grid = get_param_grid(model_name)
        grid_search = GridSearchCV(clf, param_grid=param_grid)
        grid_search.fit(X, y)
        clf = grid_search.best_estimator_
    else:
        clf.fit(X, y)

    # save the classifier
    file_name = "../classifiers/extended/" + pitcher_name + ".pkl"
    joblib.dump(clf, file_name)

def load(pitcher_name):
    file_name = "../classifiers/extended/" + pitcher_name + ".pkl"
    clf = joblib.load(file_name)
    return clf

def predict(clf, data, pitcher_name, model_name, save):
    X = data[0]
    y = data[1]
    predictions = clf.predict(X)
    eval(clf, y, predictions)
    if save:
        write_results(y, predictions, pitcher_name, model_name)
    return scores(clf, X, y)

def get_param_grid(model):
    if (model == 'nn'):
      param_grid = {'hidden_layer_sizes': [(10, ), (25, ), (50, ), (75, ), (100, )],
                    'activation' : ['relu', 'logistic', 'tanh', 'identity'],
                    'solver' : ['lbfgs', 'sgd', 'adam'],
                    'alpha' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                    'batch_size' : [200,400,600,800,1000],                    
                    }
    elif (model == 'rf'):
      param_grid = {'n_estimators': [10, 500],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    "max_depth": [3, 5, None],
                    "min_samples_leaf": [1, 3, 5],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    "n_jobs": [-1]
                    }
    elif (model == 'svm'):
      param_grid = {'C' : [1e-2, 1e-1, 1, 1e1, 1e2],
                    'kernel': ['poly', 'rbf', 'sigmoid'],
                    'degree': [3, 5, 7, 9],
                    'gamma': [1e-2, 1e-1, 1, 1e1]
                    }
    return param_grid


def eval(clf, act, pred):
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(act, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(act, pred))


def scores(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=2)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return np.mean(scores)


def clear_txt_files(model_names):
    for model_name in model_names:
        file_name = "../classifiers/results/" + model_name + "-results.txt" 
        with open(file_name, "w") as f:
            f.write("RESULTS:\n\n")

    file_name = "../classifiers/results/best-results.txt" 
    with open(file_name, "w") as f:
        f.write("RESULTS:\n\n")


def write_results(act, pred, pitcher_name, model_name):
    file_nmae = "../classifiers/results/" + model_name + "-results.txt" 
    with open(file_nmae, "a") as f:
      f.write(pitcher_name + ":\n")
      f.write("----------------------------------------------------\n")
      f.write(metrics.classification_report(act,pred))
      f.write("\n\n")


def write_best_scores(scores, pitcher_name):
    with open("../classifiers/results/best-results.txt", "a") as f:
        model_name, score, clf = scores[0]
        f.write(pitcher_name + ":\n")
        f.write("----------------------------------------------------\n")
        f.write("model: %s\n" % model_name)
        f.write("score: %f\n" %float(score))
        f.write("\n\n")
        file_name = "../classifiers/best/" + pitcher_name + ".pkl"
        joblib.dump(clf, file_name)

if __name__ == "__main__":
    main()