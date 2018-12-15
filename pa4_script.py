"""
This file contains only the best classifier. Consult PA4_script.ipynb for the different models we tried out.
Both files laod all their functions from pa4_functions.py. 

Use this file as follows: 
python pa4_script.py "path_to_train" "path_to_test" "name_of_prediction_file" "path_to_gold_standard"

If no arguments are provided it defaults to:
python pa4_script.py "train.json.txt" "test.json.txt" "predictions.txt" "test_labeled.json.txt"
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import sys
from sklearn.metrics import classification_report

from pa4_functions import load_data, print_stats, ExractSimpleFeatures, extractSegments, evaluateCV
from pa4_functions import evaluateCV_check, printLabelsToFile, printNMostInformative, gridSearchCV, plot_pos_neg_extreme_coefficients
from pa4_functions import *

def useClassifier(clf, encoder, train_features, train_labels, test_features, filename):
    avg = evaluateCV(clf, encoder, train_features, train_labels)
    print("Average: %.3f" % avg)
    evaluateCV_check(clf, train_features, train_labels)

    clf.fit(train_features, train_labels)   
    test_label_predicted = clf.predict(test_features)

    printLabelsToFile(encoder, test_label_predicted, filename)
    print("printed to " + '"' + filename + '"')

if __name__ == '__main__':
    path_to_train = None
    path_to_test = None
    path_to_predictions = None
    path_to_gold_standard = None
    if sys.argv and len(sys.argv) > 2:
        path_to_train = sys.argv[1]
        path_to_test = sys.argv[2]
        path_to_predictions = sys.argv[3]
        path_to_gold_standard = sys.argv[4]
    if not path_to_train:
        path_to_train = 'train.json.txt'
    if not path_to_test:
        path_to_test = 'test.json.txt'
    if not path_to_predictions:
        path_to_predictions = 'predictions.txt'
    if not path_to_gold_standard:
        path_to_gold_standard = 'test_labeled.json.txt'

    ###########################################################################################
    # 2. LOAD DATA
    ###########################################################################################
    train_data, train_labels = load_data(path_to_train, verbose=False)
    test_data, test_labels = load_data(path_to_test, verbose=False)

    ###########################################################################################
    # 2. EXTRACT FEATURES and LABELS
    ###########################################################################################
    # LABELS
    le = LabelEncoder()
    train_labels_onehot =le.fit_transform(train_labels)

    # FEATURES - TRAIN & TEST
    train_data_featurized = ExractSimpleFeatures(train_data, verbose=False)
    test_data_featurized = ExractSimpleFeatures(test_data, verbose=False)

    train_data_middle_segment = extractSegments(train_data, False, True, False)
    train_data_all_segments = extractSegments(train_data, True, True, True)
    test_data_middle_segment = extractSegments(test_data, False, True, False)
    test_data_all_segments = extractSegments(test_data, True, True, True)

    ###########################################################################################
    # 3. BUILD PIPELINES
    ###########################################################################################
    #clf = make_pipeline(DictVectorizer(), LogisticRegression())
    #clf2 = make_pipeline(TfidfVectorizer(), LogisticRegression())
    #clf3 = make_pipeline(TfidfVectorizer(ngram_range=(0, 3), analyzer='char'), LogisticRegression())
    clf = make_pipeline(CountVectorizer(ngram_range=(1, 2)), LogisticRegression(multi_class='ovr', solver='lbfgs'))

    ###########################################################################################
    # 4. USE PIPELINES
    ###########################################################################################
    #useClassifier(clf, le, train_data_featurized, train_labels_onehot, test_data_featurized, 'predictions_clf.txt')
    #useClassifier(clf2, le, train_data_middle_segment, train_labels_onehot, test_data_middle_segment, 'predictions_clf2.txt')
    print("#########################################################################################")
    print("Best Classifier")
    useClassifier(clf, le, train_data_all_segments, train_labels_onehot, test_data_all_segments, path_to_predictions)

    #########################################################################################
    # 4. PRINT MOST IMPORTANT FEATURES
    #########################################################################################

    print("Top features used to predict: ")
    #printNMostInformative(clf, le, 3, 'dictvectorizer')
    printNMostInformative(clf, le, 3, 'countvectorizer')

    print("The following are more detailed graphs of the most positive and the most negative features for the Regression: ")
    plot_pos_neg_extreme_coefficients(clf, 'countvectorizer', le)

    gold_standard_labels = get_gold_standard_labels(path_to_gold_standard)
    our_predictions_labels = get_predicted_labels(path_to_predictions)

    print(classification_report(gold_standard_labels, our_predictions_labels))
