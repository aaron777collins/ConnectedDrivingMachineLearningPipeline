# prepares the data, trains the pipelines and classifies the data based on the classifiers

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from MDataClassifier import MDataClassifier
CLASSIFIER_INSTANCES = [RandomForestClassifier(
), DecisionTreeClassifier(), KNeighborsClassifier()]


class MClassifierPipeline:
    def __init__(self, train_X, train_Y, test_X, test_Y, classifier_instances=CLASSIFIER_INSTANCES):

        # create MDataClassifier instances for each classifier
        self.classifiers = []
        for classifier_instance in classifier_instances:
            self.classifiers.append(
                MDataClassifier(classifier_instance, train_X, train_Y, test_X, test_Y))

    def getClassifiers(self):
        return self.classifiers

    def train(self):
        for mClassifier in self.classifiers:
            mClassifier.train()

    def test(self):
        for mClassifier in self.classifiers:
            mClassifier.classify()

    def test_training_set(self):
        for mClassifier in self.classifiers:
            mClassifier.classify_train()

    # gets accuracy, precision, recall and f1 score for each classifier and associates it with the classifier
    def calc_classifier_results(self):
        self.results = []

        for mClassifier in self.classifiers:
            self.results.append((mClassifier, mClassifier.get_results()))

        return self

    def get_classifier_results(self):
        return self.results



