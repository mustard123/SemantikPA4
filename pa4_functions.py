import gzip
import numpy as np
import random
import os
import json

from collections import Counter, namedtuple
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

def load_data(file, verbose=True):
    PairExample = namedtuple('PairExample', 'entity_1, entity_2, snippet')
    Snippet = namedtuple('Snippet', 'left, mention_1, middle, mention_2, right, direction')

    f = open(file, 'r', encoding='utf-8')
    data = []
    labels = []
    for i, line in enumerate(f):
        instance = json.loads(line)
        if i == 0:
            if verbose:
                print('json example:')
                print(instance)
        # 'relation, entity_1, entity_2, snippet' fileds for each example
        # 'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'], instance['entity_2'], [])
        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'], snippet['mention_1'],
                                        snippet['middle'],
                                        snippet['mention_2'], snippet['right'],
                                        snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)
            except:
                print(instance)
        if i == 0:
            if verbose:
                print('\nexample transformed as a named tuple:')
                print(instance_tuple)
        data.append(instance_tuple)
        labels.append(instance['relation'])
    return data, labels

# Extract two simple features
def ExractSimpleFeatures(data, verbose=True):
    featurized_data = []
    for instance in data:
        featurized_instance = {'mid_words':'', 'distance':np.inf}
        for s in instance.snippet:
            if len(s.middle.split()) < featurized_instance['distance']:
                featurized_instance['mid_words'] = s.middle
                featurized_instance['distance'] = len(s.middle.split())
        featurized_data.append(featurized_instance)
    if verbose:
        print(len(data))
        print(len(featurized_data))
        print(data[0])
        print(featurized_data[0])
    return featurized_data

# Extract all middle segments
# One instance may contain multiple snippets
def extractSegments(data, left, middle, right):
    all_segments = []
    for instance in data:
        combined_segments = ""
        for snippet in instance.snippet:
            if left:
                combined_segments += snippet.left + " "
            if middle:
                combined_segments += snippet.middle + " "
            if right:
                combined_segments += snippet.right + " "
        all_segments.append(combined_segments)
    return all_segments

# Statistics over relations
def print_stats(labels):
    labels_counts = Counter(labels)
    print('{:20s} {:>10s} {:>10s}'.format('', '', 'rel_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('relation', 'examples', '/all_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    for k,v in labels_counts.items():
        print('{:20s} {:10d} {:10.2f}'.format(k, v, v /len(labels)))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    print('{:20s} {:10d} {:10.2f}'.format('Total', len(labels), len(labels) /len(labels)))

def print_statistics_header():
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        'relation', 'precision', 'recall', 'f-score', 'support'))
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))


def print_statistics_row(rel, result):
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format(rel, *result))


def print_statistics_footer(avg_result):
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format('macro-average', *avg_result))


def macro_average_results(results):
    avg_result = [np.average([r[i] for r in results.values()]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results.values()]))
    return avg_result


def average_results(results):
    avg_result = [np.average([r[i] for r in results]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results]))
    return avg_result


def evaluateCV(classifier, label_encoder, X, y, verbose=True):
    results = {}
    for rel in label_encoder.classes_:
        results[rel] = []
    if verbose:
        print_statistics_header()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_index, test_index in kfold.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            classifier.fit(X_train, y_train)
            pred_labels = classifier.predict(X_test)
            stats = precision_recall_fscore_support(y_test, pred_labels, beta=0.5)
            # print(stats)
            for rel in label_encoder.classes_:
                rel_id = label_encoder.transform([rel])[0]
                # print(rel_id,rel)
                stats_rel = [stat[rel_id] for stat in stats]
                results[rel].append(stats_rel)
        for rel in label_encoder.classes_:
            results[rel] = average_results(results[rel])
            if verbose:
                print_statistics_row(rel, results[rel])
    avg_result = macro_average_results(results)
    if verbose:
        print_statistics_footer(avg_result)
    return avg_result[2]  # return f_0.5 score as summary statistic


def evaluateCV_check(classifier, X, y, verbose=True):    
    # A check for the average F1 score
    f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')

    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=kfold, scoring = f_scorer)
    print("\nCross-validation scores (StratifiedKFold): ", scores)
    print("Mean cv score (StratifiedKFold): ", scores.mean())

# Print labels to file
def printLabelsToFile(encoder, labels, fileName):
    labels_recoded = encoder.inverse_transform(labels)
    with open(fileName, "w", encoding="utf-8") as f:
        for lbl in labels_recoded:
            f.write(str(lbl) + '\n')

# Feature analisys - print N most informative
# !! Make changes in this function when you change the pipleine!!
def printNMostInformative(classifier, label_encoder, N, pipeline_preprocessor):
    """Prints features with the highest coefficient values, per class"""
    feature_names = classifier.named_steps[pipeline_preprocessor].get_feature_names()

    coef = classifier.named_steps['logisticregression'].coef_
    print(coef.shape)
    for rel in label_encoder.classes_:
        rel_id = label_encoder.transform([rel])[0]
        coef_rel = coef[rel_id]
        coefs_with_fns = sorted(zip(coef_rel, feature_names))
        top_features = coefs_with_fns[-N:]
        print("\nClass {} best: ".format(rel))
        for feat in top_features:
            print(feat)

# GridSearch with cross validation
# verbose can be set to 0 for no prints, 1 for little prints, and 2 for detailed prints
def gridSearchCV(pipeline, parameters, features, labels, verbose):
    f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')
    gridSearch = GridSearchCV(pipeline, parameters, scoring=f_scorer, cv=5, verbose=verbose)
    gridSearch.fit(features, labels)
    means = gridSearch.cv_results_['mean_test_score']
    stds = gridSearch.cv_results_['std_test_score']
    parameters = gridSearch.cv_results_['params']
    print("Best: ")
    print("%0.3f (+/-%0.03f) for %r" % (means[gridSearch.best_index_], 
                                        stds[gridSearch.best_index_] * 2, parameters[gridSearch.best_index_]))
    print("\nGrid: ")
    for mean, std, params in zip(means, stds, parameters):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    return gridSearch.best_params_

#plots the most negative and th most positive coefficients
def plot_pos_neg_extreme_coefficients(classifier, vectorizer, le, top_features=20):
    class_names = le.classes_
    for name in class_names:
        print(name)
        rel_id = le.transform([name])[0]
        coef = classifier.named_steps['logisticregression'].coef_[rel_id]
        feature_names = classifier.named_steps[vectorizer].get_feature_names()

        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)

        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.title(name)
        plt.show()


def get_gold_standard_labels(path_to_gold_standard):
    data, gold_standard_labels = load_data(path_to_gold_standard, verbose=False)
    return gold_standard_labels


def get_predicted_labels(path_to_prediction_file):
    with open(path_to_prediction_file) as f:
        content = f.readlines()
    labels = [line.strip() for line in content]
    return labels



