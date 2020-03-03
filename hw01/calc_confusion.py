import numpy as np

input = [[0.9, 0.1, 0.],
         [1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0., 1., 0.],
         [0., 0., 1.],
         [0.1, 0.9, 0.],
         [0.9, 0.1, 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0., 0., 1.],
         [0., 0., 1.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0.9, 0., 0.1],
         [0., 1., 0.],
         [0., 0.5, 0.5],
         [0., 1., 0.],
         [0., 0.5, 0.5],
         [0., 1., 0.],
         [0., 0., 1.],
         [0., 0.5, 0.5],
         [0., 0., 1.],
         [1., 0., 0.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0.7, 0.3, 0.],
         [1., 0., 0.],
         [1., 0., 0.]]

lbls = [0, 0, 1, 0, 2, 0, 2, 1, 2, 1, 0, 1, 0, 2, 0, 2, 2, 2, 0, 2, 0, 2,
        2, 1, 1, 1, 2, 2, 2, 0, 0, 2, 1, 0, 0]


def precision(mtrx):
    true_positives, true_negatives, false_negatives, false_positives = mtrx
    return true_positives / (true_positives + false_positives)


def falsepositiverate(mtrx):
    true_positives, true_negatives, false_negatives, false_positives = mtrx
    return false_positives / (false_positives + true_negatives)


def recall(mtrx):
    true_positives, true_negatives, false_negatives, false_positives = mtrx
    return true_positives / (true_positives + false_negatives)


def f1_score(mtrx):
    return 2 / (precision(mtrx) + recall(mtrx))


def get_confusion_matrix(labels, probas, target, thre):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(0, len(labels)):
        if labels[i] == target:
            positive = True
        else:
            positive = False

        proba = probas[i][target]

        if proba > thre and positive:
            true_positives += 1
        if proba > thre and not positive:
            false_positives += 1
        if proba <= thre and positive:
            false_negatives += 1
        if proba <= thre and not positive:
            true_negatives += 1
    return (true_positives, true_negatives, false_negatives, false_positives)

def roc(lbls, probas, tgt):
    thresholds = np.linspace(0, 1, 20)

    x = np.zeros(len(thresholds))
    y = np.zeros(len(thresholds))

    idx = 0
    for t in thresholds:
        mtrx = get_confusion_matrix(labels=lbls, probas=probas, target=tgt, thre=t)

        tpr = recall(mtrx)
        fpr = falsepositiverate(mtrx)
        x[idx] = fpr
        y[idx] = tpr

        idx+=1

    return (x,y)


print(roc(lbls, input, 1 ))