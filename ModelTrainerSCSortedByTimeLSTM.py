from cgi import test
from datetime import datetime as dt
import os.path as path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from EasyMLLib.helper import Helper
from DataCleaner import DataCleaner
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from matplotlib import pyplot as plt

from EasyMLLib.DataSplitter import DataSplitter
from EasyMLLib.logger import Logger
from EasyMLLib.ModelSaver import ModelSaver
from EasyMLLib.CSVWriter import CSVWriter

from DataGatherer import DataGatherer

from tensorflow import keras
from tensorflow.keras import layers

# DIB_NAME = "DIB1"
# DATASETS_PATH = path.join(DIB_NAME, 'dataRaw', 'ID')
# SELECTED_DATA_SET_PATH = path.join("Single", "ET")
# CONCAT_FILE_PATH = DIB_NAME
# CONCAT_FILE_NAME = "concat-data"
# CONCAT_FILE_EXT = ".csv"
# CONCAT_FULLPATH_WITHOUT_EXT = path.join(CONCAT_FILE_PATH, CONCAT_FILE_NAME)

MODEL_FILE_NAME_BEGINNING = "model-"
MODEL_EXT = ".model"
MODEL_NAME = "LSTM"

BATCH_SIZE = 128
# 3 times the number of cols in the data
EPOCHS = 40

# TESTS = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']
TESTS = [accuracy_score, precision_score, recall_score, f1_score]
TESTS_WITH_SAMPLE_NAMES = []
for test in TESTS:
    TESTS_WITH_SAMPLE_NAMES.append(f"train-{test.__name__}")
    # TESTS_WITH_SAMPLE_NAMES.append(f"val-{test.__name__}")
    TESTS_WITH_SAMPLE_NAMES.append(f"test-{test.__name__}")
#     TESTS_WITH_SAMPLE_NAMES.append(f"train-{test.__name__}-time")
#     TESTS_WITH_SAMPLE_NAMES.append(f"val-{test.__name__}-time")
#     TESTS_WITH_SAMPLE_NAMES.append(f"test-{test.__name__}-time")

# TESTS_WITH_SAMPLE_NAMES.append(f"train")
# TESTS_WITH_SAMPLE_NAMES.append(f"val")
# TESTS_WITH_SAMPLE_NAMES.append(f"test")
TESTS_WITH_SAMPLE_NAMES.append(f"train-time")
# TESTS_WITH_SAMPLE_NAMES.append(f"val-time")
TESTS_WITH_SAMPLE_NAMES.append(f"test-time")

CSV_COLUMNS = ["Model", "Total Compile Time",
               "Total Sample Size", "Compile Time Per Sample"]
CSV_COLUMNS.extend(TESTS_WITH_SAMPLE_NAMES)

CSV_FORMAT = {CSV_COLUMNS[i]: i for i in range(len(CSV_COLUMNS))}


# PARAM
OVERWRITE_MODEL = True


class ModelTrainerSortedByTime:

    def get_uncompiled_model(self):
        LSTMModel = keras.Sequential()

        LSTMModel.add(layers.Input(13))

        # Add an Embedding layer expecting input vocab of size (inputted), and
        # output embedding dimension of size 64.
        #  60000, 1000
        #  13000000
        #  12491236
        # 12500000
        LSTMModel.add(layers.Embedding(input_dim=13000000, output_dim=1000, input_length=13))

        # Add a LSTM layer with 128 internal units.
        # Changing to different amount
        LSTMModel.add(layers.LSTM(128))

        # Add a Dense layer with 2 units.
        # sigmoid is the proper activation function for binary classification
        LSTMModel.add(layers.Dense(1, activation='sigmoid'))

        LSTMModel.summary()
        return LSTMModel

# Q: what is the proper optimizer for binary classification with tensorflow?
# A: https://stackoverflow.com/questions/44467379/what-is-the-proper-optimizer-for-binary-classification-with-tensorflow

    def get_compiled_model(self):
        LSTMModel = self.get_uncompiled_model()
        LSTMModel.compile(optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.Accuracy(),
                 keras.metrics.Precision(),
                 keras.metrics.Recall()])
        return LSTMModel


    def main(self):

        id = "SC-concat"

        print(f"Creating Logger for model with id:{id}")
        self.logger = Logger( f"Model-{id}-merged-sorted-LSTM.txt")
        self.csvWriter = CSVWriter(f"Models-{id}-merged-sorted-LSTM.csv", CSV_COLUMNS)

        self.logger.log("Getting ground truth file...[1/4]")
        groundTruthData = DataGatherer.gatherData(DataGatherer.DATA_PATH, DataGatherer.GROUND_TRUTH_FILENAME, DataGatherer.REFINED_DATA_PATH, DataGatherer.RAW_FILE_NAME)

        self.logger.log("Converting test json files to csv files and expanding columns...[2/4]")
        populatedTestFiles = DataCleaner.getPopulatedData()
        self.logger.log("Cleaning data...[3/4]")
        cleanedTestFiles = DataCleaner.getCleanData()

        self.logger.log("Done!")
        self.logger.log("First result:")
        self.logger.log(str(cleanedTestFiles[0].head(5)))
        self.logger.log(str(cleanedTestFiles[0].shape))
        # for cleanedTestFile in cleanedTestFiles:
        #         print(cleanedTestFile.head(2))

        self.logger.log("Merging... [4/4]")
        data = DataCleaner.getCleanMergedSortedData()
        self.logger.log("Done!")
        self.logger.log(str(data.head(5)))
        self.logger.log(str(data.shape))


        self.logger.log("Quick stats on clean, merged and sorted data")
        Helper.quickDfStat(data)

        self.logger.log("Getting Data Sets..")
        startTime = dt.now()
        # features, answers = DataSplitter(classifierName='isAttacker').getAllFeaturesAndAnswers(data)
        features, answers = [data.drop(['isAttacker'], axis=1), pd.DataFrame(data, columns=['isAttacker'])]
        # X_train, Y_train, X_val, Y_val, X_test, Y_test = DataSplitter(classifierName='isAttacker').getTrainValTestSplit(data)

        shape_80 = int(features.shape[0]*0.8)-1
        X_train = features.iloc[:shape_80, :]
        Y_train = answers.iloc[:shape_80, :]
        X_test = features.iloc[shape_80:, :]
        Y_test = answers.iloc[shape_80:, :]




        self.logger.log( f"Time elapsed: (hh:mm:ss:ms) {dt.now()-startTime}")

        self.logger.log( "Quick stats on features and answers for the train-val-test split")
        Helper.quickDfArrStat([X_train, Y_train])
        Helper.quickDfArrStat([X_test, Y_test])

        self.logger.log( "Verifying the features and answers for the sets add up")

        # self.logger.log("Verifying X..")
        featureArr = []
        for df in [X_train, X_test]:
            val = round(len(df.index)/len(features.index), 3)
            featureArr.append(val)
            # self.logger.log(f"{val}")

        # self.logger.log("Verifying Y..")
        answerArr = []
        for df in [Y_train, Y_test]:
            val = round(len(df.index)/len(answers.index), 3)
            answerArr.append(val)
            # self.logger.log(f"{val}")

        self.logger.log("Adding up X")
        sum = 0
        for x in featureArr:
            sum += x
        self.logger.log(f"Sum: {sum}")
        self.logger.log("Adding up Y")
        sum = 0
        for y in answerArr:
            sum += y
        self.logger.log(f"Sum: {sum}")

        self.logger.log("Building", MODEL_NAME)

        model_name = MODEL_FILE_NAME_BEGINNING + \
            f"{MODEL_NAME}-" + f"{id}" + MODEL_EXT
        modelCompileTime = (dt.now()-dt.now())

        # model = ModelSaver[StackingClassifier]().readModel(model_name)

        # if(not model or OVERWRITE_MODEL):
        self.logger.log(f"Building Model on: {MODEL_NAME}")

        startTime = dt.now()

        [model, history] = self.buildModel(X_train, Y_train, self.get_compiled_model())
        modelCompileTime = (dt.now()-startTime)
        self.logger.log(
            f"Time elapsed: (hh:mm:ss:ms) {modelCompileTime}")

        self.logger.log("History: ", str(history.history))

        # self.logger.log(f"Saving Model as: {model_name}")
        # startTime = dt.now()
        # ModelSaver().saveModel(model, model_name)
        # self.logger.log(
        #     f"Time elapsed: (hh:mm:ss:ms) {dt.now()-startTime}")

        row = [" "] * len(CSV_COLUMNS)
        row[CSV_FORMAT["Model"]] = MODEL_NAME
        row[CSV_FORMAT["Total Compile Time"]] = modelCompileTime
        row[CSV_FORMAT["Total Sample Size"]] = len(X_train.index)
        row[CSV_FORMAT["Compile Time Per Sample"]
            ] = modelCompileTime.total_seconds() / len(X_train.index)

        self.logger.log(f"Possible tests:", metrics.SCORERS.keys())

        self.logger.log("Testing model on Train")
        startTime = dt.now()
        y_pred = model.predict(X_train)
        timeElapsed = dt.now()-startTime
        self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"train-time"]] = timeElapsed.total_seconds() / \
            len(X_train.index)

        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_train, y_pred)
            else:
                res = test_type(Y_train, y_pred, average='macro')
            self.logger.log(f"train-{test_type.__name__}:", res)
            row[CSV_FORMAT[f"train-{test_type.__name__}"]] = res

        # self.logger.log("Testing model on val")
        # startTime = dt.now()
        # y_pred = model.predict(X_val)
        # timeElapsed = dt.now()-startTime
        # self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        # row[CSV_FORMAT[f"val-time"]] = timeElapsed.total_seconds() / \
        #     len(X_val.index)
        # for test_type in TESTS:
        #     res = None
        #     if (test_type.__name__ == accuracy_score.__name__):
        #         res = test_type(Y_val, y_pred)
        #     else:
        #         res = test_type(Y_val, y_pred, average='macro')
        #     self.logger.log(f"val-{test_type.__name__}:", res)
        #     row[CSV_FORMAT[f"val-{test_type.__name__}"]] = res

        #     self.logger.log("Testing model on Val")
        # for test_type in TESTS:
        #     startTime = dt.now()
        #     res = cross_val_score(
        #         model, X_val, Y_val, cv=5, scoring=test_type)
        #     self.logger.log(f"Tested {test_type}", res)
        #     timeElapsed = dt.now()-startTime
        #     self.logger.log(
        #         f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        #     row[CSV_FORMAT[f"val-{test_type}"]] = res
        #     row[CSV_FORMAT[f"val-{test_type}-time"]
        #         ] = timeElapsed.total_seconds() / len(X_val.index)

        self.logger.log("Testing model on test")
        startTime = dt.now()
        y_pred = model.predict(X_test)
        timeElapsed = dt.now()-startTime
        self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"test-time"]] = timeElapsed.total_seconds() / \
            len(X_test.index)
        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_test, y_pred)
            else:
                res = test_type(Y_test, y_pred, average='macro')
            self.logger.log(f"test-{test_type.__name__}:", res)
            row[CSV_FORMAT[f"test-{test_type.__name__}"]] = res

        # self.logger.log("Testing model on Test")
        # for test_type in TESTS:
        #     startTime = dt.now()
        #     res = cross_val_score(
        #         model, X_test, Y_test, cv=5, scoring=test_type)
        #     self.logger.log(f"Tested {test_type}", res)
        #     timeElapsed = dt.now()-startTime
        #     self.logger.log(
        #         f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        #     row[CSV_FORMAT[f"test-{test_type}"]] = res
        #     row[CSV_FORMAT[f"test-{test_type}-time"]
        #         ] = timeElapsed.total_seconds() / len(X_test.index)

        self.csvWriter.addRow(row)

    # FEATURE TESTING CODE BELOW

        # get importance
        # importance = model.feature_importances_

        # # Summarize feature importance
        # for f,s in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (f,s))

        # # plot feature importance
        # fig, ax = plt.subplots(figsize=(30, 10))
        # ax.bar([x for x in range(len(importance))], importance)
        # print(features.columns.values)
        # ax.set_xticks([x for x in range(len(importance))])
        # ax.set_xticklabels(labels=features.columns.values)
        # #plt.show()
        # plt.savefig(path.join("Figures", "feature-test-"+ classifierNames[i] +"-"+ datetime.now().strftime(R"%m-%d-%Y, %H-%M-%S") + ".png"), format='png', dpi=200)

    def buildModel(self, features: pd.DataFrame, answers: pd.DataFrame, model) -> Tuple:
        # from tutorial: https://machinelearningmastery.com/calculate-feature-importance-with-python/

        # fit the model
        history = model.fit(features, answers, batch_size=BATCH_SIZE, epochs=EPOCHS)

        return [model, history]


if __name__ == "__main__":
    ModelTrainerSortedByTime().main()
