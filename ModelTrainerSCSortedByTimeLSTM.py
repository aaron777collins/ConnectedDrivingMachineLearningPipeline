from cgi import test
from datetime import datetime as dt
import os.path as path
from pathlib import Path
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
from tensorflow import math as mathtf
import tensorflow as tf

import os

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

MODELS_FOLDER = "models"

# BATCH_SIZE = 128
BATCH_SIZE = 5000
# 3 times the number of cols in the data
EPOCHS = 10

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

NUM_INPUTS = 21

# PARAM
OVERWRITE_MODEL = True

class ModelTrainerSCSortedByTimeLSTM:

    @staticmethod
    def round_0_or_1(number) -> float:
        if (number < 0.5):
            return 0.0
        else:
            return 1.0

    def get_uncompiled_model(self):
        LSTMModel = keras.Sequential()

        LSTMModel.add(layers.Input(NUM_INPUTS))

        # Add an Embedding layer expecting input vocab of size (inputted), and
        # output embedding dimension of size 64.
        #  60000, 1000
        #  13000000
        #  12491236
        # 12500000                             12500000
        #                                      1033395
        # 1250000
        # rcvTime         5.418034e+04
        # sendTime        5.418034e+04
        # sender          6.123000e+03
        # senderPseudo    1.061235e+06 # = 1061235 -> round up -> 1070000
        # isAttacker      1.000000e+00
        # posX            1.395715e+03
        # posY            1.359091e+03
        # spdX            1.816491e+01
        # spdY            1.793936e+01
        # aclX            4.501414e+00
        # aclY            4.499957e+00
        # hedX            1.000000e+00
        # hedY            1.000000e+00
        # posXNeg         0.000000e+00
        # posYNeg         0.000000e+00
        # spdXNeg         1.000000e+00
        # spdYNeg         1.000000e+00
        # aclXNeg         1.000000e+00
        # aclYNeg         1.000000e+00
        # hedXNeg         1.000000e+00
        # hedYNeg         1.000000e+00
        # dtype: float64

        # output layer is supposed to be the 4th root of the number of categories
        # Even if we assume that worst case we have 1070000 categories, it would be 32.16
        # Another website suggests dividing the number of unique categories by 2 and putting a max of 50
        # Thus, to be safe, we will use the number 50 because it's always better to be safe.

        LSTMModel.add(layers.Embedding(input_dim=1070000, output_dim=50, input_length=NUM_INPUTS))

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
        modelIDStr = f"Model-{id}-merged-sorted-LSTM-v2"
        self.logger = Logger( modelIDStr + ".txt")
        self.csvWriter = CSVWriter(modelIDStr + ".csv", CSV_COLUMNS)

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
        data = DataCleaner.getCleanMergedDataWithoutMsgID()
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

        Path(MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

        model = None

        modelPathStr = os.path.join(MODELS_FOLDER, modelIDStr)

        try:
            model = keras.models.load_model(modelPathStr)
        except IOError:

            self.logger.log(f"Building Model on: {MODEL_NAME}")

            startTime = dt.now()

            [model, history] = self.buildModel(X_train, Y_train, self.get_compiled_model())
            modelCompileTime = (dt.now()-startTime)
            self.logger.log(
                f"Time elapsed: (hh:mm:ss:ms) {modelCompileTime}")

            self.logger.log("History: ", str(history.history))

            self.logger.log(f"Saving Model as: {model_name}")
            startTime = dt.now()
            model.save(modelPathStr)
            self.logger.log(
                f"Time elapsed: (hh:mm:ss:ms) {dt.now()-startTime}")

        row = [" "] * len(CSV_COLUMNS)
        row[CSV_FORMAT["Model"]] = MODEL_NAME
        row[CSV_FORMAT["Total Compile Time"]] = modelCompileTime
        row[CSV_FORMAT["Total Sample Size"]] = len(X_train.index)
        row[CSV_FORMAT["Compile Time Per Sample"]
            ] = modelCompileTime.total_seconds() / len(X_train.index)

        self.logger.log(f"Possible tests:", metrics.SCORERS.keys())

        roundNums = lambda x: ModelTrainerSCSortedByTimeLSTM.round_0_or_1(x)

        self.logger.log("Testing model on Train")
        startTime = dt.now()
        y_pred = roundNums(model.predict(X_train))

        timeElapsed = dt.now()-startTime
        self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"train-time"]] = timeElapsed.total_seconds() / \
            len(X_train.index)

        # convert (n,1) to (n,) and then map to round
        y_pred_transposed_rounded = list(map(ModelTrainerSCSortedByTimeLSTM.round_0_or_1, np.transpose(y_pred)[0]))

        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_train, y_pred_transposed_rounded)
            else:
                res = test_type(Y_train, y_pred_transposed_rounded, average='macro')
            self.logger.log(f"train-{test_type.__name__}:", res)
            row[CSV_FORMAT[f"train-{test_type.__name__}"]] = res

        self.logger.log("Testing model on test")
        startTime = dt.now()
        y_pred = roundNums(model.predict(X_test))
        timeElapsed = dt.now()-startTime
        self.logger.log(f"Time elapsed: (hh:mm:ss:ms) {timeElapsed}")
        row[CSV_FORMAT[f"test-time"]] = timeElapsed.total_seconds() / \
            len(X_test.index)

        # convert (n,1) to (n,) and then map to round
        y_pred_transposed_rounded = list(map(ModelTrainerSCSortedByTimeLSTM.round_0_or_1, np.transpose(y_pred)[0]))

        for test_type in TESTS:
            res = None
            if (test_type.__name__ == accuracy_score.__name__):
                res = test_type(Y_test, y_pred_transposed_rounded)
            else:
                res = test_type(Y_test, y_pred_transposed_rounded, average='macro')
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
    ModelTrainerSCSortedByTimeLSTM().main()
