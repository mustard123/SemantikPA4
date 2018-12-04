from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from pa4_functions import load_data, print_stats, ExractSimpleFeatures, extractSegments, evaluateCV, evaluateCV_check, printLabelsToFile, printNMostInformative

def useClassifier(clf, encoder, train_features, train_labels, test_features, filename):
    avg = evaluateCV(clf, encoder, train_features, train_labels)
    print("Average: %.3f" % avg)
    evaluateCV_check(clf, train_features, train_labels)

    clf.fit(train_features, train_labels)   
    test_label_predicted = clf.predict(test_features)

    printLabelsToFile(encoder, test_label_predicted, filename)
    print("printed to " + '"' + filename + '"')

if __name__ == '__main__':

    ###########################################################################################
    # 2. LOAD DATA
    ###########################################################################################
    train_data, train_labels = load_data('train.json.txt', verbose=False)
    test_data, test_labels = load_data('test.json.txt', verbose=False)

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
    clf3 = make_pipeline(TfidfVectorizer(ngram_range=(0, 3), analyzer='char'), LogisticRegression())

    ###########################################################################################
    # 4. USE PIPELINES
    ###########################################################################################
    #useClassifier(clf, le, train_data_featurized, train_labels_onehot, test_data_featurized, 'predictions_clf.txt')
    #useClassifier(clf2, le, train_data_middle_segment, train_labels_onehot, test_data_middle_segment, 'predictions_clf2.txt')
    print("#########################################################################################")
    print("clf3")
    useClassifier(clf3, le, train_data_middle_segment, train_labels_onehot, test_data_middle_segment, 'predictions_clf3.txt')

    #########################################################################################
    # 4. PRINT MOST IMPORTANT FEATURES
    #########################################################################################

    print("Top features used to predict: ")
    #printNMostInformative(clf, le, 3, 'dictvectorizer')
    printNMostInformative(clf3, le, 3, 'tfidfvectorizer')
    
