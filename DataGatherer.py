import pandas as pd
from pandas import DataFrame
import os.path as path
import json
import pathlib
import glob


class DataGatherer:
    DATA_PATH = path.join("data", "DataReplay_1416", "VeReMi_50400_54000_2022-9-11_19.11.57", "VeReMi_50400_54000_2022-9-11_19_11_57") ##what does this do?
    GROUND_TRUTH_FILENAME = "traceGroundTruthJSON-14.json"
    REFINED_DATA_PATH = path.join("data", "refinedData", "VeReMi_50400_54000_2022-9-11_ConstSpeedOffset")
    RAW_FILE_NAME= "groundtruth.csv"

    @staticmethod
    def gatherData(dataPath, groundTruthFileName, refinedDataPath, refinedDataFileName) -> DataFrame:
        pathlib.Path(refinedDataPath).mkdir(parents=True, exist_ok=True)
        refinedDataFilePath = path.join(refinedDataPath, refinedDataFileName)

        if(path.isfile(refinedDataFilePath)):
            return pd.read_csv(refinedDataFilePath)


        groundTruthFile = open(path.join(dataPath, groundTruthFileName), "r")
        newData = groundTruthFile.readline()

        fileData = []

        while len(newData) > 0:
            fileData.append(json.loads(newData))
            newData = groundTruthFile.readline()

        dataframeData = DataFrame.from_records(fileData)
        dataframeData.to_csv(refinedDataFilePath, index=False)

        return dataframeData
