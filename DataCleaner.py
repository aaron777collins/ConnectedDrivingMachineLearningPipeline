import string
from typing import List
from DataGatherer import DataGatherer
import pandas as pd
from pandas import DataFrame
import os.path as path
import json
import pathlib
import glob
import numpy as np

class DataCleaner:

    POPULATED_DATA_PATH = path.join("data", "populatedData", "VeReMi_50400_54000_2022-9-11_19.11.57")
    CLEANED_DATA_PATH = path.join("data", "cleanedData", "VeReMi_50400_54000_2022-9-11_19.11.57")
    IS_ATTACKER_DATA_PATH = path.join("data", "isAttacker", "VeReMi_50400_54000_2022-9-11_19.11.57")
    IS_ATTACKER_DATA_FILE = path.join(IS_ATTACKER_DATA_PATH, "isattacker.txt")

    @staticmethod
    def populateData(df: DataFrame, fileName: string) -> DataFrame:

        maliciousFilesIDs = DataCleaner.getMaliciousFiles()

        df.drop(df[df['type'] == 2].index, inplace=True)
        df['receiverID'] = fileName.split("-")[1]
        try:
            df["isAttacker"] = np.where(df["sender"].astype(int).isin(maliciousFilesIDs), True, False)
        except Exception as e:
            print("Warning: Error was caught:")
            print(e)
            print("File ", fileName, "had issues being expanded and will likely be removed since it contains ONLY type 2 messages (Couldn't find the 'sender' columnn)")


        posDf = None
        try:
            posDf = DataFrame(df["pos"].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.split(",").to_list(), columns=["posX", "posY", "posZ"])
            posDf["posX"] = posDf["posX"].astype('float64')
            posDf["posY"] = posDf["posY"].astype('float64')
        except Exception as e:
            # print(e)
            posDf = DataFrame(df["pos"].to_list(), columns=["posX", "posY", "posZ"])
            # print("Trying as regular array instead:")
            if (len(posDf.index) == 0):
                print("Error converting pos, size of index is 0")

        spdDf = None
        try:
            spdDf = DataFrame(df["spd"].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.split(",").to_list(), columns=["spdX", "spdY", "spdZ"])
            spdDf["spdX"] = spdDf["spdX"].astype('float64')
            spdDf["spdY"] = spdDf["spdY"].astype('float64')
        except Exception as e:
            # print(e)
            spdDf = DataFrame(df["spd"].to_list(), columns=["spdX", "spdY", "spdZ"])
        aclDf = None
        try:
            aclDf = DataFrame(df["acl"].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.split(",").to_list(), columns=["aclX", "aclY", "aclZ"])
            aclDf["aclX"] = aclDf["aclX"].astype('float64')
            aclDf["aclY"] = aclDf["aclY"].astype('float64')
        except Exception as e:
            # print(e)
            aclDf = DataFrame(df["acl"].to_list(), columns=["aclX", "aclY", "aclZ"])
        hedDf = None
        try:
            hedDf = DataFrame(df["hed"].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.split(",").to_list(), columns=["hedX", "hedY", "hedZ"])
            hedDf["hedX"] = hedDf["hedX"].astype('float64')
            hedDf["hedY"] = hedDf["hedY"].astype('float64')
        except Exception as e:
            # print(e)
            hedDf = DataFrame(df["hed"].to_list(), columns=["hedX", "hedY", "hedZ"])


        posDf.drop("posZ", axis=1, inplace=True)
        df = df.join(posDf)
        spdDf.drop("spdZ", axis=1, inplace=True)
        df = df.join(spdDf)
        aclDf.drop("aclZ", axis=1, inplace=True)
        df = df.join(aclDf)
        hedDf.drop("hedZ", axis=1, inplace=True)
        df = df.join(hedDf)

        df.drop("pos", axis=1, inplace=True)
        df.drop("spd", axis=1, inplace=True)
        df.drop("acl", axis=1, inplace=True)
        df.drop("hed", axis=1, inplace=True)
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
        df.drop('receiverID', axis=1, inplace=True)
        df.drop('type', axis=1, inplace=True)
        # df.drop('senderPseudo', axis=1, inplace=True)
        # df.drop('sender', axis=1, inplace=True)
        return df

    @staticmethod
    def getMaliciousFiles() -> List[int]:
        maliciousFileIDs = []

        pathlib.Path(DataCleaner.IS_ATTACKER_DATA_PATH).mkdir(parents=True, exist_ok=True)

        dataFiles = glob.glob(DataGatherer.REFINED_DATA_PATH + "/*.csv")
        filteredFiles = list(filter(filterGroundTruthPath, dataFiles))
        if path.isfile(DataCleaner.IS_ATTACKER_DATA_FILE):
            with open(DataCleaner.IS_ATTACKER_DATA_FILE, 'r') as fp:
                for line in fp:
                    # remove linebreak from a current name
                    # linebreak is the last character of each line
                    item = line[:-1]

                    # add current item to the list
                    maliciousFileIDs.append(int(item))
        else:
            for file in filteredFiles:
                # Initial Scan for malicious files:
                # Remember which files are malicious and store it in a file
                # df['receiverID'] = fileName.split("-")[1]
                # df['isAttacker'] = True if fileName.split("-")[3] != "A0" else False
                baseName = path.basename(file)
                if (baseName.split("-")[3] != "A0"):
                    maliciousFileIDs.append(int(baseName.split("-")[1]))

            with open(DataCleaner.IS_ATTACKER_DATA_FILE, 'w') as fp:
                for item in maliciousFileIDs:
                    # write each item on a new line
                    fp.write("%d\n" % item)
        return maliciousFileIDs

    @staticmethod
    def getCleanData() -> List[DataFrame]:
        cleanedTestFiles = []
        dataFiles = glob.glob(DataCleaner.POPULATED_DATA_PATH + "/*.csv")
        pathlib.Path(DataCleaner.CLEANED_DATA_PATH).mkdir(parents=True, exist_ok=True)
        for file in dataFiles:
            cleanedDataFilePath = path.join(DataCleaner.CLEANED_DATA_PATH, path.basename(file))

            if path.isfile(cleanedDataFilePath):
                cleanedTestFiles.append(pd.read_csv(cleanedDataFilePath))
            else:
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



