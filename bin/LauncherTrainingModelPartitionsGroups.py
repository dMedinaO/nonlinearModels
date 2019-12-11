import argparse
import glob
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pathInput", help="Path where is all data set created using clustering partitions", required=True)
parser.add_argument("-r", "--response", help="name of column with response values in dataset", required=True)
parser.add_argument("-k", "--kValueData", type=int, help="Value for cross validation, this value most be higher or equal 2", default=2)
parser.add_argument("-t", "--type", type=int, help="Type of response in dataset: 1. class, 2. prediction", choices=[1,2], required=True)

args = parser.parse_args()

performanceTraining = []
totalExamples = 0

#una vez procesada la data... para cada elemento en formato csv en el path response... lo leemos
listFiles = glob.glob(args.pathInput+"*.csv")

for files in listFiles:
    print "Training DataSet: ", files

    ID_dataset = files.split("/")[-1].split(".csv")[0]
    print ID_dataset

    print "To Create dir %s", ID_dataset
    command = "mkdir -p %s%s" % (args.pathInput, ID_dataset)
    os.system(command)

    #armamos el comando para hacer la ejecucion...
    if args.type == 1:#Classification model

        dataSet = files
        pathResponse = args.pathInput+ID_dataset+"/"
        response = args.response

        command = "python LauncherTrainingModelPartitions.py -d %s -p %s -r %s" % (dataSet, pathResponse, response)
        print "Training model %s" % ID_dataset
        print command

    else:#prediction model
        dataSet = files
        pathResponse = args.pathInput+ID_dataset+"/"
        response = args.response

        command = "python LauncherTrainingModelPrediction.py -d %s -p %s -r %s" % (dataSet, pathResponse, response)
        print "Training model %s" % ID_dataset
        print command

    os.system(command)

    try:
        #procesamos las medidas de desempeno...
        dataFrame = pd.read_csv(pathResponse+"summarySelectedModel.csv")

        performanceGroup = []

        #agregamos el largo
        dataSetInput = pd.read_csv(dataSet)
        performanceGroup.append(len(dataSetInput))
        totalExamples+=len(dataSetInput)

        #agregamos el ID del grupo
        performanceGroup.append(ID_dataset)

        #obtenemos los promedios...
        if args.type == 1:
            performanceGroup.append(np.mean(dataFrame['Accuracy']))#2
            performanceGroup.append(np.mean(dataFrame['Recall']))#3
            performanceGroup.append(np.mean(dataFrame['Precision']))#4
            performanceGroup.append(np.mean(dataFrame['F1']))#5
        else:
            performanceGroup.append(np.mean(dataFrame['R_Score']))
            performanceGroup.append(np.mean(dataFrame['Pearson']))
            performanceGroup.append(np.mean(dataFrame['Spearman']))
            performanceGroup.append(np.mean(dataFrame['Kendalltau']))

        #agregamos la data a la matriz
        performanceTraining.append(performanceGroup)
    except:
        pass
#obtenemos las performance ponderadas del entrenamiento
p1=0
p2=0
p3=0
p4=0
for element in performanceTraining:

    p1+= float(element[0])*float(element[2])/float(totalExamples)
    p2+= float(element[0])*float(element[3])/float(totalExamples)
    p3+= float(element[0])*float(element[4])/float(totalExamples)
    p4+= float(element[0])*float(element[5])/float(totalExamples)


print "Performance Traing weighted:"
if args.type == 1:
    print "Accuracy: ", p1
else:
    print "R_Score: ", p1
    print "Pearson: ", p2
    print "Spearman: ", p3
    print "Kendalltau: ", p4
