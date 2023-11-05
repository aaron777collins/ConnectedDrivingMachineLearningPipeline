# %%

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
from DataCleanerConstSpeedOffset import DataCleanerConstSpeedOffset
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from EasyMLLib.DataSplitter import DataSplitter
from EasyMLLib.logger import Logger
from EasyMLLib.ModelSaver import ModelSaver
from EasyMLLib.CSVWriter import CSVWriter

from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick

import shap

from DataGatherer import DataGatherer

import os

from MClassifierPipeline import MClassifierPipeline

# DIB_NAME = "DIB1"
# DATASETS_PATH = path.join(DIB_NAME, 'dataRaw', 'ID')
# SELECTED_DATA_SET_PATH = path.join("Single", "ET")
# CONCAT_FILE_PATH = DIB_NAME
# CONCAT_FILE_NAME = "concat-data"
# CONCAT_FILE_EXT = ".csv"
# CONCAT_FULLPATH_WITHOUT_EXT = path.join(CONCAT_FILE_PATH, CONCAT_FILE_NAME)

MODEL_FILE_NAME_BEGINNING = "model-"
MODEL_EXT = ".model"
MODEL_NAME = "MLConstSpeedOffset"

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

# Was 21 but removed messageID
NUM_INPUTS = 20

# PARAM
OVERWRITE_MODEL = True

class ModelTrainerMLSpeedOffsetConstSCSortedByTime:

    @staticmethod
    def round_0_or_1(number) -> float:
        if (number < 0.5):
            return 0.0
        else:
            return 1.0

    
    def main(self):

        id = "SC-concat"

        print(f"Creating Logger for model with id:{id}")
        modelIDStr = f"CONSTSPEED-Model-{id}-merged-sorted-LSTM-v2"
        self.logger = Logger( modelIDStr + ".txt") # type: ignore
        self.csvWriter = CSVWriter(modelIDStr + ".csv", CSV_COLUMNS) # type: ignore

        self.logger.log("Getting ground truth file...[1/4]")
        DataPath = path.join("data", "SpeedOffset", "SPEEDCONSTVeReMi_50400_54000_2022-9-11_18_23_0")
        groundTruthData = DataGatherer.gatherData(DataPath, "traceGroundTruthJSON-14.json", DataCleanerConstSpeedOffset.REFINED_DATA_PATH, DataGatherer.RAW_FILE_NAME)

        self.logger.log("Converting test json files to csv files and expanding columns...[2/4]")
        populatedTestFiles = DataCleanerConstSpeedOffset.getPopulatedData()
        self.logger.log("Cleaning data...[3/4]")
        cleanedTestFiles = DataCleanerConstSpeedOffset.getCleanData()

        self.logger.log("Done!")
        self.logger.log("First result:")
        self.logger.log(str(cleanedTestFiles[0].head(5)))
        self.logger.log(str(cleanedTestFiles[0].shape))
        # for cleanedTestFile in cleanedTestFiles:
        #         print(cleanedTestFile.head(2))

        self.logger.log("Merging... [4/4]")
        data = DataCleanerConstSpeedOffset.getCleanMergedSortedDataWithoutMsgID()
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
        self.logger.log( f"THE FEATURES ARE: {X_train.columns}" )
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

        # using the mclassiferpipeline to train on the data
        model_instances = [RandomForestClassifier(), KNeighborsClassifier(), DecisionTreeClassifier()]

        mPipeline = MClassifierPipeline(X_train, Y_train, X_test, Y_test, classifier_instances=model_instances)
        mPipeline.train()
        mPipeline.test_training_set()
        mPipeline.test()
        results = mPipeline.calc_classifier_results().get_classifier_results()

        for classifier,result in results:

            row = [" "] * len(CSV_COLUMNS)
            row[CSV_FORMAT["Model"]] = classifier.classifier.__class__.__name__
            row[CSV_FORMAT["Total Compile Time"]] = classifier.elapsed_train_time
            row[CSV_FORMAT["Total Sample Size"]] = len(X_train.index) # type: ignore
            row[CSV_FORMAT["Compile Time Per Sample"]
                ] = classifier.elapsed_train_time / len(X_train.index)


            self.logger.log(f"Time elapsed: (ss) {classifier.elapsed_prediction_time}")
            row[CSV_FORMAT[f"train-time"]] = classifier.elapsed_prediction_time / \
                len(X_train.index)

            y_pred = classifier.predicted_train_results
            # convert (n,1) to (n,) and then map to round
            print(y_pred)
            #y_pred_transposed_rounded = list(map(ModelTrainerMLPosOffsetConstSCSortedByTime.round_0_or_1, np.transpose(y_pred)[0]))

            for test_type in TESTS:
                res = None
                if (test_type.__name__ == accuracy_score.__name__):
                    res = test_type(Y_train, y_pred)
                else:
                    res = test_type(Y_train, y_pred, average='macro')
                self.logger.log(f"train-{test_type.__name__}:", res)
                row[CSV_FORMAT[f"train-{test_type.__name__}"]] = res

            y_pred = classifier.predicted_results
            self.logger.log(f"Time elapsed: (ss) {classifier.elapsed_prediction_time}")
            row[CSV_FORMAT[f"test-time"]] = classifier.elapsed_prediction_time / \
                len(X_test.index)

            y_pred = classifier.predicted_results
            # convert (n,1) to (n,) and then map to round
            print(y_pred)
            #y_pred_transposed_rounded = list(map(ModelTrainerMLPosOffsetConstSCSortedByTime.round_0_or_1, np.transpose(y_pred)[0]))

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

            ##XAI STUFF HERE
            answerList = ["isAttacker", "notAttacker"]
            featureList = ['rcvTime', 'sendTime', 'sender', 'senderPseudo', 'posX', 'posY', 'spdX','spdY', 'aclX', 'aclY', 'hedX', 'hedY', 'posXNeg', 'posYNeg', 'spdXNeg','spdYNeg', 'aclXNeg', 'aclYNeg', 'hedXNeg', 'hedYNeg']

            classifierPredictionFunction = classifier.classifier.predict_proba

            explainer_lime = LimeTabularExplainer(X_train.values, feature_names = featureList, class_names= answerList, mode='classification')
           
            i= 47
            exp_lime = explainer_lime.explain_instance(X_test.values[i], predict_fn=classifierPredictionFunction, num_features=len(featureList))

            # exp_lime = submodular_pick.SubmodularPick(explainer_lime, X_train.values, predict_fn=classifierPredictionFunction, num_features=len(featureList), num_exps_desired=10)
            # [exp.show_in_notebook() for exp in exp_lime.sp_explanations]

            explanationHTMLpath = f'Outputs/EXPLANATION{classifier.classifier.__class__.__name__}{modelIDStr}.html'
            exp_lime.save_to_file(explanationHTMLpath)

         
            #Shap stuff

            #shap.initjs()

            # exp_shap = shap.TreeExplainer(skLearnClassifierObj)

            # shap_values = exp_shap.shap_values(X_train)

            #shapImagePath = f'Outputs/shap_summary_{classifier.classifier.__class__.__name__}.pdf'
            # shap.force_plot(exp_shap, shap_values, X_train, matplotlib=True)
            # plt.savefig(f"sample.jpg", bbox_inches='tight')


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

        randomForestC = results[0][0].classifier
        
        object_type = type(randomForestC)
        print("The type of randomForestC is:", object_type)

        shap.initjs()

        explainer = shap.TreeExplainer(randomForestC)

        object_type2 = type(explainer)
        print("The type of explainer is:", object_type2)

        object_type3 = type(X_test)
        print("The type of X_test is:", object_type3)

        print(X_test.head())
        print(X_test.head())

        howManyRows = 400
        shap_values = explainer.shap_values(X_test.head(howManyRows))

        object_type4 = type(shap_values)
        print("The type of shap_values is:", object_type4)

        object_type5 = type(features)
        print(features.head())
        print("The type of features is:", object_type5)

        #Print out shap values for others 
        shap.summary_plot(shap_values=shap_values, features=features.head(howManyRows), class_names=['Is Attacker', 'Is Not Attacker'])

        shap.summary_plot(shap_values=shap_values[0], features=features.head(howManyRows), class_names=['Is Attacker', 'Is Not Attacker'], plot_type="bar")

        shap.summary_plot(shap_values[0], X_test.head(howManyRows), plot_type="dot")

        expectedValue = explainer.expected_value
        shap.decision_plot(expectedValue[0], shap_values[0], X_test.columns) # type: ignore

        
if __name__ == "__main__":
    ModelTrainerMLSpeedOffsetConstSCSortedByTime().main()


# %%
