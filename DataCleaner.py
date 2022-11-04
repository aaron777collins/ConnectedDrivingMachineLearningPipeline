import string
from typing import List
from DataGatherer import DataGatherer
import pandas as pd
from pandas import DataFrame
import os.path as path
import json
import pathlib
import glob

class DataCleaner:

    POPULATED_DATA_PATH = path.join("data", "populatedData", "VeReMi_50400_54000_2022-9-11_19.11.57")
    CLEANED_DATA_PATH = path.join("data", "cleanedData", "VeReMi_50400_54000_2022-9-11_19.11.57")

    @staticmethod
    def populateData(df: DataFrame, fileName: string) -> DataFrame:
        df.drop(df[df['type'] == 2].index, inplace=True)
        df['receiverID'] = fileName.split("-")[1]
        df['isAttacker'] = True if fileName.split("-")[3] != "A0" else False
        posDf = None
        try:
            posDf = DataFrame(df["pos"].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.split(",").to_list(), columns=["posX", "posY", "posZ"])
            posDf["posX"] = posDf["posX"].astype('float64')
            posDf["posY"] = posDf["posY"].astype('float64')
        except:
            posDf = DataFrame(df["pos"].to_list(), columns=["posX", "posY", "posZ"])
        posDf.drop("posZ", axis=1, inplace=True)
        df = df.join(posDf)
        df.drop("pos", axis=1, inplace=True)
        return df

    @staticmethod
    def getPopulatedData() -> List[DataFrame]:
        dataFiles = glob.glob(DataGatherer.DATA_PATH + "/*.json")
        filteredFiles = list(filter(filterGroundTruthPath, dataFiles))

        populatedTestFiles = []

        for file in filteredFiles:
            pathlib.Path(DataCleaner.POPULATED_DATA_PATH).mkdir(parents=True, exist_ok=True)
            populatedDataFilePath = path.join(DataCleaner.POPULATED_DATA_PATH, path.basename(file).replace(".json", ".csv"))
            if path.isfile(populatedDataFilePath):
                return pd.read_csv(populatedDataFilePath)


            populatedTestFile = DataCleaner.populateData(DataGatherer.gatherData(path.dirname(file), path.basename(file), DataGatherer.REFINED_DATA_PATH, path.basename(file).replace(".json", ".csv")), path.basename(file).replace(".json", ".csv"))
            if (len(populatedTestFile.index) == 0):
                continue
            populatedTestFiles.append(populatedTestFile)
            populatedTestFile.to_csv(populatedDataFilePath, index=False)

        return populatedTestFiles

    @staticmethod
    def cleanData(df: DataFrame) -> DataFrame:
        df.drop('pos_noise', axis=1, inplace=True)
        df.drop('spd_noise', axis=1, inplace=True)
        df.drop('acl_noise', axis=1, inplace=True)
        df.drop('hed_noise', axis=1, inplace=True)
        df.drop('senderPseudo', axis=1, inplace=True)
        df.drop('sender', axis=1, inplace=True)
        return df

    @staticmethod
    def getCleanData() -> List[DataFrame]:
        dataFiles = glob.glob(DataCleaner.POPULATED_DATA_PATH + "/*.csv")

        cleanedTestFiles = []

        for file in dataFiles:
            pathlib.Path(DataCleaner.CLEANED_DATA_PATH).mkdir(parents=True, exist_ok=True)
            cleanedDataFilePath = path.join(DataCleaner.CLEANED_DATA_PATH, path.basename(file))
            if path.isfile(cleanedDataFilePath):
                return pd.read_csv(cleanedDataFilePath)

            cleanedTestFile = None
            cleanedTestFile = DataCleaner.cleanData(pd.read_csv(file))
            cleanedTestFile.to_csv(cleanedDataFilePath, index=False)
            cleanedTestFiles.append(cleanedTestFile)
        return cleanedTestFiles




def filterGroundTruthPath(pathname):
    return not ("groundtruth" in pathname.lower())

if (__name__ == "__main__"):
    print("Getting ground truth file...[1/3]")
    groundTruthData = DataGatherer.gatherData(DataGatherer.DATA_PATH, DataGatherer.GROUND_TRUTH_FILENAME, DataGatherer.REFINED_DATA_PATH, DataGatherer.RAW_FILE_NAME)

    print("Converting test json files to csv files and expanding columns...[2/3]")
    populatedTestFiles = DataCleaner.getPopulatedData()
    print("Cleaning data...[3/3]")
    cleanedTestFiles = DataCleaner.getCleanData()

    print("Done!")
    print("First result:")
    print(cleanedTestFiles[0].head(5))
    # for cleanedTestFile in cleanedTestFiles:
    #         print(cleanedTestFile.head(2))



