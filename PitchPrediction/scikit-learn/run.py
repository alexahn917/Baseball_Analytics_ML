#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8
from __future__ import division
import csv
import pickle
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn import decomposition
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer as DV
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def main():

    menu = "         Pitch Prediction Model          \n"\
           "=========================================\n"\
           "(1) train & test pitchers                \n"\
           "(2) predict next pitch ball              \n"\
           "(3) visualize the data                   \n"\
           "(4) exit                                 \n"\

    while (True):
        print(menu)
        option = raw_input("Enter option: ")
        if option == "1":
            train_test()
        elif option == "2":
            load_predict()
        elif option == "3":
            visualize()
        elif option == "4":
            exit(0)
        else:
            print("Wrong option, try again.")


def train_test():
    with open("../pitchers_0.txt", "r") as f:
        names = f.read().split('\n')
    #names = ["Clayton Kershaw"]
    model_names = ['svm', 'nn', 'rf']

    clear_txt_files(model_names)
    for pitcher_name in names:
        train_data, test_data = readCSV(pitcher_name, True)
        scores = []
        for model_name in model_names:
            train(train_data, pitcher_name, model_name, grid_search=False)
            clf = load(pitcher_name)
            scores.append((model_name, predict(clf, test_data, pitcher_name, model_name, save=True), clf))
        pitch_types = np.unique(test_data[1])
        write_summary(scores, pitcher_name, len(pitch_types), len(train_data[1]), pitch_types)
        scores.sort(key=lambda tup:tup[1], reverse=True)
        write_best_scores(scores, pitcher_name, test_data[1])    

def load_predict():
    pitcher_name = raw_input("Enter pitcher's name: ")
    clf = load(pitcher_name)
    predict_pitches(clf, pitcher_name)

def visualize():
    pitcher_name = raw_input("Enter pitcher's name: ")
    data = readFullCSV(pitcher_name)
    col_names = data.dtypes.index
    y = data['pitch_type']
    print(data.shape, y.shape)

    # convert to integer labels
    labels = np.unique(y)
    labels_int = np.ndarray(shape=(len(y),))
    for i in range(len(labels)):
        labels_int[np.where(y == labels[i])] = i
    
    data['pitch_type'] = labels_int
    
    plot_PCA(data, labels_int, pitcher_name, True)
    #plot_corr_mat(data.T, col_names)

def plot_PCA(X, labels, pitcher_name, drawThirdDim):
    X = X[:] - np.mean(X[:])
    pca = decomposition.PCA(n_components=X.iloc[0,:].size)
    pca.fit(X)
    X_pca = pca.transform(X)
    E_vectors = pca.components_.T
    E_values = pca.explained_variance_
    print("Explained variance with 2 eigan vectors: %f%%" %np.sum(pca.explained_variance_ratio_[:2]))

    # 2D plot 
    plt.scatter(X_pca[:,0], X_pca[:,2], s=1, c=labels, marker='o')
    plt.xlabel("First Pricinple Component")
    plt.ylabel("Second Pricinple Component")
    plt.title("PCA visualization on %s\'s pitches"%pitcher_name)
    plt.show()

    # 3D plot 
    if (drawThirdDim):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=1, c=labels, marker='o')
        ax2.set_xlabel('First Pricinple Component')
        ax2.set_ylabel('Second Pricinple Component')
        ax2.set_zlabel('Third Pricinple Component')
        plt.title("PCA visualization on %s\'s pitches"%pitcher_name)
        plt.show()

def plot_corr_mat(X, col_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    corr_mat = np.corrcoef(X)
    print(corr_mat.shape)
    np.fill_diagonal(corr_mat, 0)
    plt.imshow(corr_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ax.set_yticks(np.arange(X.shape[0]))
    ax.set_yticklabels(col_names, rotation='horizontal', fontsize=8)
    ax.set_xticks(np.arange(X.shape[0]))
    ax.set_xticklabels(col_names, rotation=70, fontsize=8)
    plt.title("Correlation Matrix of Pitch Factors")
    plt.show()    


def readCSV(pitcher_name, populate=True):
    train_csv_file = '../ETL pipeline/CSV/raw/' + pitcher_name + '_train' + '.csv'
    test_csv_file = '../ETL pipeline/CSV/raw/' + pitcher_name + '_test' + '.csv'

    X_train = pd.DataFrame.from_csv(train_csv_file, index_col=None)
    y_train = X_train['pitch_type']
    X_train.drop('pitch_type', axis = 1, inplace=True)

    X_test = pd.DataFrame.from_csv(test_csv_file, index_col=None)
    y_test = X_test['pitch_type']
    X_test.drop('pitch_type', axis = 1, inplace=True)

    # populate data to equalize distribution
    if populate:
        X_train, y_train = populateData(X_train, y_train)
    return [X_train, y_train], [X_test, y_test]

def readFullCSV(pitcher_name):
    full_csv_file = '../ETL pipeline/CSV/full/' + pitcher_name + '.csv'
    return pd.DataFrame.from_csv(full_csv_file, index_col=None)

def populateData(X_train, y_train):
    labels = np.unique(y_train)
    labels_counts = []
    for label in labels:
        labels_counts.append((label,np.sum(y_train == label)))
    labels_counts.sort(key=lambda tup:tup[1], reverse=True)
    
    max_counts = labels_counts[0][1]
    # sample rest of the smaller labeled datasets and append
    for i in range(1, len(labels_counts)):
        sample_size = max_counts - labels_counts[i][1]
        curr_label = labels_counts[i][0]
        #print("Label %s is %d many short in numbers"%(labels_counts[i][0], max_counts - labels_counts[i][1]))
        indicies = [i for i, x in enumerate(y_train==curr_label) if x]
        sampled_idx = np.random.choice(indicies, sample_size, replace=True)
        X_train = X_train.append(X_train.iloc[sampled_idx], ignore_index=True)
        y_train = y_train.append(pd.Series([curr_label] * sample_size), ignore_index=True)
    
    perm_idx = np.random.permutation(y_train.index)
    X_train = X_train.reindex(perm_idx)
    y_train = y_train.reindex(perm_idx)
    return X_train, y_train


def train(data, pitcher_name, model_name, grid_search):
    X = data[0]
    y = data[1]
    n_samples = len(X)

    if len(np.unique(y)) is 1:
        print(pitcher_name)
        print(np.unique(n_samples))
        print(n_samples)
        exit(1)
    if model_name == 'svm':
        clf = svm.SVC(decision_function_shape='ovr', gamma='auto', kernel='rbf')
    elif model_name == 'nn':
        clf = MLPClassifier()
    elif model_name == 'rf':
        clf = RandomForestClassifier()
    
    # grid search
    if grid_search:
        param_grid = get_param_grid(model_name)
        grid_search = GridSearchCV(clf, param_grid=param_grid)#, scoring='f1_macro')
        print("\nfitting..")
        print(param_grid)
        grid_search.fit(X, y)
        print("\nfitting done!")
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
      param_grid = {'hidden_layer_sizes': [(100, ), (300, ), (500, ), (800, )],
                    'activation' : ['relu', 'logistic'],#, 'tanh', 'identity'],
                    'solver' : ['sgd', 'adam', 'lbfgs'], 
                    'alpha' : [0.001, 0.005, 0.01, 0.1],
                    #'batch_size' : [400,600,800],
                    }
    elif (model == 'rf'):
      param_grid = {'n_estimators': [30, 50, 70, 100],
                    #'n_estimators': [10, 30, 50],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    "max_depth": [3, 5, 7],
                    #"min_samples_leaf": [1, 3, 5],
                    "bootstrap": [True, False],
                    "criterion": ["gini", "entropy"],
                    }
    elif (model == 'svm'):
        param_grid = {'C' : [5, 7, 9, 12],
                    'kernel': ['rbf'], #'poly'],#, 'sigmoid'],
                    'gamma': [5e-3, 1e-2, 1e-1],
                    #'max_iter' : [100, 200, 300]
                    #'degree': [3, 5, 7, 9],
                    }
    return param_grid


def eval(clf, act, pred):
    print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(act, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(act, pred))


def scores(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return np.mean(scores)


def clear_txt_files(model_names):
    for model_name in model_names:
        file_name = "../classifiers/results/" + model_name + "-results.txt" 
        with open(file_name, "w") as f:
            f.write("RESULTS:\n\n")

    file_name = "../classifiers/results/best-results.txt" 
    with open(file_name, "w") as f:
        f.write("Best classifier results::\n\n")
    file_name = "../classifiers/results/summary.txt" 
    with open(file_name, "w") as f:
        f.write("Prediction Accuracy Summary:\n\n")


def write_results(act, pred, pitcher_name, model_name):
    file_nmae = "../classifiers/results/" + model_name + "-results.txt" 
    with open(file_nmae, "a") as f:
      f.write(pitcher_name + ":\n")
      f.write("----------------------------------------------------\n")
      f.write(metrics.classification_report(act,pred))
      f.write("\n\n")


def write_best_scores(scores, pitcher_name, data):
    with open("../classifiers/results/best-results.txt", "a") as f:
        model_name, score, clf = scores[0]
        f.write(pitcher_name + ":\n")
        f.write("----------------------------------------------------\n")
        f.write("model: %s\n" % model_name)
        f.write(str(clf))
        f.write("\n\n classes: %s" %(np.unique(data)))
        f.write("\n score: %f\n" %float(score))
        f.write("\n\n")
        file_name = "../classifiers/best/" + pitcher_name + ".pkl"
        joblib.dump(clf, file_name)

def write_summary(scores, pitcher_name, label_size, train_size, types):
    with open("../classifiers/results/summary.txt", "a") as f:
        table = "====================================================\n"\
                "  Pitcher: {0:12s}                                  \n\n"\
                "  Pitch Types:{6}                                   \n"\
                "  label_size:{4}   train_size:{5}                   \n"\
                "----------------------------------------------------\n"\
                "  SVM         NeuN       RanF                       \n"\
                "  =====       =====      =====                      \n"\
                "  {1:.3f}       {2:.3f}      {3:.3f}                \n"\
                "====================================================\n\n"\
                .format(pitcher_name, float(scores[0][1]), float(scores[1][1]), float(scores[2][1]),\
                label_size, train_size, types)
        
        f.write(table)

def predict_pitches(clf, pitcher_name):
    #Clayton Kershaw
    csv_file = '../ETL pipeline/CSV/raw/' + pitcher_name + '_test' + '.csv'
    df = pd.DataFrame.from_csv(csv_file, index_col=None)
    labels = np.unique(df['pitch_type'])
    fields = df.dtypes.index
    fields = fields.delete(0)

    menu = "\n          Next Pitch Prediction          \n"\
           "=========================================\n\n"\
           "please enter the following...\n"\

    labels_menu = "\n<Pitch Labels>\n"
    for i in range(len(labels)):
        labels_menu += ("(%s) %s\n" %(str(i), labels[i]))

    # fill in known entries
    x = pd.Series(index=fields)
    x.fillna(0, inplace=True)
    x['pitch_rl'] = df['pitch_rl'][0]

    # game situations
    inning = int(raw_input("inning: ") or 1)
    balls = int(raw_input("balls: ") or 0)
    strikes = int(raw_input("strikes: ") or 0)
    out = int(raw_input("outs: ") or 0)
    on_1b = int(raw_input("runner on 1st base (1 if yes): ") or 0)
    on_2b = int(raw_input("runner on 2nd base (1 if yes): ") or 0)
    on_3b = int(raw_input("runner on 3rd base (1 if yes): ") or 0)
    score_diff = int(raw_input("score differencial (+ for pitcher): ") or 0)

    # pitcher
    pitcher_at_home = int(raw_input("pitcher at home (1 if true): ") or 0)
    era = float(raw_input("pitcher's current ERA: ") or 0)
    pitcher_wins = int(raw_input("# of wins for pitcher: ") or 0)
    pitcher_losses = int(raw_input("# of losses for pitcher: ") or 0)

    bat_order, bat_rl, rbi, avg, hr, batter_wins, batter_losses = get_batter_input()

    x['inning'] = inning
    x['balls'] = balls
    x['strikes'] = strikes
    x['out'] = out
    x['on_1b'] = on_1b
    x['on_2b'] = on_2b
    x['on_3b'] = on_3b
    x['score_diff'] = score_diff
    x['pitcher_at_home'] = pitcher_at_home
    x['era'] = era
    x['pitcher_wins'] = pitcher_wins
    x['pitcher_losses'] = pitcher_losses
    x['bat_order'] = bat_order
    x['bat_rl'] = bat_rl
    x['rbi'] = rbi
    x['avg'] = avg
    x['hr'] = hr
    x['batter_wins'] = batter_wins
    x['batter_losses'] = batter_losses
    x['prev_pitch_type0'] = 1
    x['prevprev_pitch_type0'] = 1

    results_menu = "Select the next result:\n"\
           "(1) ball               \n"\
           "(2) strike             \n"\
           "(3) out                \n"\
           "(4) strike out         \n"\
           "(0) exit               \n"\

    prev = 'prev_pitch_type0'
    prevprev = 'prevprev_pitch_type0'
    while True:
        print x
        print get_predict_menu(clf, x)
        event = int(raw_input(results_menu) or 1)
        pitch = labels[int(raw_input(labels_menu) or 0)]
        x[prevprev] = 0
        x[prev] = 0
        prevprev = "prev" + prev
        prev = "prev_pitch_type"+pitch
        x[prevprev] = 1
        x[prev] = 1
        if event is 1:
            x['balls'] += 1
        elif event is 2:
            x['strikes'] += 1
        elif event is 3:
            x['out'] += 1
            bat_order, bat_rl, rbi, avg, hr, batter_wins, batter_losses = get_batter_input()
            x['bat_rl'] = bat_rl
            x['rbi'] = rbi
            x['avg'] = avg
            x['hr'] = hr
            x['batter_wins'] = batter_wins
            x['batter_losses'] = batter_losses
        elif event is 4:
            x['out'] = 0
            x['balls'] = 0
            x['strikes'] = 0
            bat_order, bat_rl, rbi, avg, hr, batter_wins, batter_losses = get_batter_input()
            x['bat_rl'] = bat_rl
            x['rbi'] = rbi
            x['avg'] = avg
            x['hr'] = hr
            x['batter_wins'] = batter_wins
            x['batter_losses'] = batter_losses
        elif event is 0:
            exit(0)
        else:
            "Wrong input, try again"
        
        #https://www.youtube.com/watch?v=Y9_6GFCWFCo //16:30

def get_predict_menu(clf, x):
    menu = "\n\n***********************************************\n\n"\
           "      Predicted next pitch: %s                 \n\n"\
           "***********************************************\n\n"\
           %clf.predict(x)[0]
    return menu

def get_batter_input():
    print("Entering the batter's statistics...")
    bat_order = int(raw_input("batting number: ") or 1)
    bat_rl = int(raw_input("batter's hand (1 if right-handed, 0 if left-handed): ") or 0)
    rbi = float(raw_input("batter's rbi: ") or 0)
    avg = float(raw_input("batter's batting avg: ") or 0)
    hr = float(raw_input("batter's homeruns: ") or 0)
    batter_wins = int(raw_input("# of wins for batter: ") or 0)
    batter_losses = int(raw_input("# of losses for batter: ") or 0)
    return bat_order, bat_rl, rbi, avg, hr, batter_wins, batter_losses

if __name__ == "__main__":
    main()