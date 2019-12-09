import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="full path and name to acces dataSet input process", required=True)
parser.add_argument("-p", "--pathResult", help="full path for save results", required=True)
parser.add_argument("-v", "--valueProportion", help="Percentage of training data set, Default 80", default=80.0, type=float)

args = parser.parse_args()

#parseamos la data
dataset = pd.read_csv(args.dataSet)
pathResult = args.pathResult
valueProportion = args.valueProportion

#validamos el porcentaje
if valueProportion>=100:
    print "Value Percentage for split must be lower than 100. It is recommendable 80%"
else:

    #obtenemos el numero de ejemplos a dividir
    numberExamples = len(dataset)
    numberTraining = int(numberExamples*valueProportion/100)
    numberTesting = numberExamples - numberTraining

    matrixTraining = []
    matrixTesting = []

    index = []
    for i in range(numberExamples):
        index.append(i)

    random.shuffle(index)#orden aleatorio de los elementos

    for i in range(numberTraining):
        row = []
        for element in dataset:
            row.append(dataset[element][i])
        matrixTraining.append(row)

    for j in range(i, numberExamples):
        row = []
        for element in dataset:
            row.append(dataset[element][j])
        matrixTesting.append(row)

    dataFrameTrainig = pd.DataFrame(matrixTraining, columns=dataset.keys())
    dataFrameTesting = pd.DataFrame(matrixTesting, columns=dataset.keys())

    print "Export data set, create file testing and training in csv format"
    dataFrameTrainig.to_csv(pathResult+"dataSetTraining.csv", index=False)
    dataFrameTesting.to_csv(pathResult+"dataSetTesting.csv", index=False)
