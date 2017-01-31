import csv
from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

def readCSV():
    csv_file = '../ETL pipeline/raw_data/Clayton Kershaw_R.csv'

    file = open(csv_file, "rb")
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

#   classifier = OneVsRestClassifier(LinearSVC(random_state=0))
#    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier = svm.SVC(decision_function_shape='ovr')
    classifier.fit(X[:n_samples / 2], Y[:n_samples / 2])
    expected = Y[n_samples / 2:]
    predicted = classifier.predict(X[n_samples / 2:])
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


if __name__ == "__main__":
    main()
