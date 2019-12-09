########################################################################
# Copyright (C) 2019  David Medina Ortiz, david.medina@cebib.cl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
########################################################################

import sys
import pandas as pd
import numpy as np
import time
import datetime
import json
import argparse

from modulesNLM.supervised_learning_predicction import AdaBoost
from modulesNLM.supervised_learning_predicction import Baggin
from modulesNLM.supervised_learning_predicction import DecisionTree
from modulesNLM.supervised_learning_predicction import Gradient
from modulesNLM.supervised_learning_predicction import knn_regression
from modulesNLM.supervised_learning_predicction import MLP
from modulesNLM.supervised_learning_predicction import NuSVR
from modulesNLM.supervised_learning_predicction import RandomForest
from modulesNLM.supervised_learning_predicction import SVR
from modulesNLM.statistics_analysis import summaryStatistic

#utils para el manejo de set de datos y su normalizacion
from modulesNLM.utils import transformDataClass
from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore
from modulesNLM.utils import summaryScanProcess
from modulesNLM.utils import responseResults
from modulesNLM.utils import encodingFeatures

#para evaluar la performance
from modulesNLM.supervised_learning_predicction import performanceData
from modulesNLM.utils import processParamsDict

#para preparar el conjunto de datos
from modulesNLM.utils import createDataSetForTraining

from joblib import dump, load

#funcion que permite obtener los indices de la data para los maximos de cada medida...
def getIndexForMaxValues(dataFrame, value, performance):

    index = 0

    for i in range(len(dataFrame)):
        if dataFrame[performance][i]>=value:
            index=i
            break
    return index

#funcion que permite calcular los estadisticos de un atributo en el set de datos, asociados a las medidas de desempeno
def estimatedStatisticPerformance(summaryObject, attribute):

    statistic = summaryObject.calculateValuesForColumn(attribute)
    row = [attribute, statistic['mean'], statistic['std'], statistic['var'], statistic['max'], statistic['min']]

    return row

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="full path and name to acces dataSet input process", required=True)
parser.add_argument("-p", "--pathResult", help="full path for save results", required=True)
parser.add_argument("-r", "--response", help="name of column with response values in dataset", required=True)

args = parser.parse_args()

#hacemos las validaciones asociadas a si existe el directorio y el set de datos
processData = responseResults.responseProcess()#parser y checks...

if (processData.validatePath(args.pathResult) == 0):

    if (processData.validateDataSetExist(args.dataSet) == 0):

        #recibimos los parametros desde la terminal...
        dataSet = pd.read_csv(args.dataSet)
        pathResponse = args.pathResult
        response = args.response

        #valores iniciales
        start_time = time.time()
        inicio = datetime.datetime.now()
        iteracionesCorrectas = 0
        iteracionesIncorrectas = 0

        #procesamos el set de datos para obtener la columna respuesta y la matriz de datos a entrenar
        target = dataSet[response]
        del dataSet[response]

        #procesamos la data de interes, asociada a la codificacion de las variables categoricas y la normalizacion del conjunto de datos
        #transformamos la clase si presenta atributos discretos
        transformData = transformDataClass.transformClass(target)
        target = transformData.transformData

        #ahora transformamos el set de datos por si existen elementos categoricos...
        #transformDataSet = transformFrequence.frequenceData(data)
        #dataSetNewFreq = transformDataSet.dataTransform
        encoding = encodingFeatures.encodingFeatures(dataSet, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet
        #ahora aplicamos el procesamiento segun lo expuesto
        applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
        data = applyNormal.dataTransform

        #obtenemos el dataset de entrenamiento y validacion, junto con los arreglos correspondientes de respuestas
        getDataProcess = createDataSetForTraining.createDataSet(data, target)
        dataSetTraining = getDataProcess.dataSetTraining
        classTraining =  getDataProcess.classTraining

        dataSetTesting = getDataProcess.dataSetTesting
        classTesting = getDataProcess.classTesting

        #generamos una lista con los valores obtenidos...
        header = ["Algorithm", "Params", "R_Score", "Pearson", "Spearman", "Kendalltau"]
        matrixResponse = []
        matrixModelData = []

        #comenzamos con las ejecuciones...

        #AdaBoost
        for loss in ['linear', 'squar', 'exponential']:
            for n_estimators in [10]:#,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec AdaBoostRegressor with %s - %d" % (loss, n_estimators)
                    AdaBoostObject = AdaBoost.AdaBoost(dataSetTraining, classTraining, n_estimators, loss)
                    AdaBoostObject.trainingMethod()

                    #obtenemos el restante de performance
                    #usamos el modelo para predecir los elementos y comparamos con respecto al valor del testing
                    predictedValues = AdaBoostObject.AdaBoostModel.predict(dataSetTraining).tolist()
                    rscore = AdaBoostObject.AdaBoostModel.score(dataSetTraining, classTraining)

                    performanceValues = performanceData.performancePrediction(classTraining, predictedValues)
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    if pearsonValue == "ERROR":
                        pearsonValue=0
                    if spearmanValue == "ERROR":
                        spearmanValue=0
                    if kendalltauValue == "ERROR":
                        kendalltauValue=0

                    params = "loss:%s-n_estimators:%d" % (loss, n_estimators)
                    row = ["AdaBoostRegressor", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                    matrixModelData.append(AdaBoostObject.AdaBoostModel)
                    break
                except:
                    iteracionesIncorrectas+=1
                    pass

        #Baggin
        for bootstrap in [True, False]:
            for n_estimators in [10]:#,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec Bagging with %s - %d" % (bootstrap, n_estimators)
                    bagginObject = Baggin.Baggin(dataSetTraining, classTraining, n_estimators, bootstrap)
                    bagginObject.trainingMethod()

                    predictedValues = bagginObject.bagginModel.predict(dataSetTraining).tolist()
                    rscore = bagginObject.bagginModel.score(dataSetTraining, classTraining)

                    performanceValues = performanceData.performancePrediction(classTraining, predictedValues)
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    if pearsonValue == "ERROR":
                        pearsonValue=0
                    if spearmanValue == "ERROR":
                        spearmanValue=0
                    if kendalltauValue == "ERROR":
                        kendalltauValue=0

                    params = "bootstrap:%s-n_estimators:%d" % (str(bootstrap), n_estimators)
                    row = ["Bagging", params, rscore, pearsonValue, spearmanValue, kendalltauValue]

                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                    matrixModelData.append(bagginObject.bagginModel)
                    break
                except:
                    iteracionesIncorrectas+=1
                    pass

        #DecisionTree
        for criterion in ['mse', 'friedman_mse', 'mae']:
            for splitter in ['best', 'random']:
                try:
                    print "Excec DecisionTree with %s - %s" % (criterion, splitter)
                    decisionTreeObject = DecisionTree.DecisionTree(dataSetTraining, classTraining, criterion, splitter)
                    decisionTreeObject.trainingMethod()

                    predictedValues = decisionTreeObject.DecisionTreeAlgorithm.predict(dataSetTraining).tolist()
                    rscore = decisionTreeObject.DecisionTreeAlgorithm.score(dataSetTraining, classTraining)

                    performanceValues = performanceData.performancePrediction(classTraining, predictedValues)
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    if pearsonValue == "ERROR":
                        pearsonValue=0
                    if spearmanValue == "ERROR":
                        spearmanValue=0
                    if kendalltauValue == "ERROR":
                        kendalltauValue=0

                    params = "criterion:%s-splitter:%s" % (criterion, splitter)
                    row = ["DecisionTree", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                    matrixModelData.append(decisionTreeObject.DecisionTreeAlgorithm)
                    break
                except:
                    iteracionesIncorrectas+=1
                    pass

        #gradiente
        for loss in ['ls', 'lad', 'huber', 'quantile']:
            for criterion in ['friedman_mse', 'mse', 'mae']:
                for n_estimators in [10]:#,50,100,200,500,1000,1500,2000]:
                    for min_samples_split in range (2, 11):
                        for min_samples_leaf in range(1, 11):
                            try:
                                print "Excec GradientBoostingRegressor with %s - %d - %d - %d" % (loss, n_estimators, min_samples_split, min_samples_leaf)
                                gradientObject = Gradient.Gradient(dataSetTraining,classTraining,n_estimators, loss, criterion, min_samples_split, min_samples_leaf)
                                gradientObject.trainingMethod()

                                predictedValues = gradientObject.GradientAlgorithm.predict(dataSetTraining).tolist()
                                rscore = gradientObject.GradientAlgorithm.score(dataSetTraining, classTraining)

                                performanceValues = performanceData.performancePrediction(classTraining, predictedValues)
                                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                                if pearsonValue == "ERROR":
                                    pearsonValue=0
                                if spearmanValue == "ERROR":
                                    spearmanValue=0
                                if kendalltauValue == "ERROR":
                                    kendalltauValue=0

                                params = "criterion:%s-n_estimators:%d-loss:%s-min_samples_split:%d-min_samples_leaf:%d" % (criterion, n_estimators, loss, min_samples_split, min_samples_leaf)
                                row = ["GradientBoostingClassifier", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                                matrixResponse.append(row)
                                iteracionesCorrectas+=1
                                print row
                                matrixModelData.append(gradientObject.GradientAlgorithm)
                                break
                            except:
                                iteracionesIncorrectas+=1
                                pass

        #knn
        for n_neighbors in range(1,11):
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                for metric in ['minkowski', 'euclidean']:
                    for weights in ['uniform', 'distance']:
                        try:
                            print "Excec KNeighborsRegressor with %d - %s - %s - %s" % (n_neighbors, algorithm, metric, weights)
                            knnObect = knn_regression.KNN_Model(dataSetTraining, classTraining, n_neighbors, algorithm, metric,  weights)
                            knnObect.trainingMethod()

                            predictedValues = knnObect.KNN_model.predict(dataSetTraining).tolist()
                            rscore = knnObect.KNN_model.score(dataSetTraining, classTraining)

                            performanceValues = performanceData.performancePrediction(classTraining, predictedValues)
                            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                            if pearsonValue == "ERROR":
                                pearsonValue=0
                            if spearmanValue == "ERROR":
                                spearmanValue=0
                            if kendalltauValue == "ERROR":
                                kendalltauValue=0

                            params = "n_neighbors:%d-algorithm:%s-metric:%s-weights:%s" % (n_neighbors, algorithm, metric, weights)
                            row = ["KNeighborsRegressor", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                            matrixResponse.append(row)
                            iteracionesCorrectas+=1
                            print row
                            matrixModelData.append(knnObect.KNN_model)
                            break
                        except:
                            iteracionesIncorrectas+=1
                            pass
        #RF
        for n_estimators in [10]:#,50,100,200,500,1000,1500,2000]:
            for criterion in ['mse', 'mae']:
                for min_samples_split in range (2, 11):
                    for min_samples_leaf in range(1, 11):
                        for bootstrap in [True, False]:
                            try:
                                print "Excec RF"
                                rf = RandomForest.RandomForest(dataSetTraining, classTraining, n_estimators, criterion, min_samples_split, min_samples_leaf, bootstrap)
                                rf.trainingMethod()

                                predictedValues = rf.randomForesModel.predict(dataSetTraining).tolist()
                                rscore = rf.randomForesModel.score(dataSetTraining, classTraining)

                                performanceValues = performanceData.performancePrediction(classTraining, predictedValues)
                                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                                if pearsonValue == "ERROR":
                                    pearsonValue=0
                                if spearmanValue == "ERROR":
                                    spearmanValue=0
                                if kendalltauValue == "ERROR":
                                    kendalltauValue=0

                                params = "n_estimators:%d-criterion:%s-min_samples_split:%d-min_samples_leaf:%d-bootstrap:%s" % (n_estimators, criterion, min_samples_split, min_samples_leaf, str(bootstrap))
                                row = ["RandomForestRegressor", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                                matrixResponse.append(row)
                                iteracionesCorrectas+=1
                                print row
                                matrixModelData.append(rf.randomForesModel)
                                break
                            except:
                                iteracionesIncorrectas+=1
                                pass

        try:
            #generamos el export de la matriz convirtiendo a data frame
            dataFrame = pd.DataFrame(matrixResponse, columns=header)

            #"R_Score", "Pearson", "Spearman", "Kendalltau"
            #obtenemos el mayor de cada medida de desempeno
            maxR_Score = max(dataFrame['R_Score'])
            maxPearson = max(dataFrame['Pearson'])
            maxSpearman = max(dataFrame['Spearman'])
            maxKendalltau = max(dataFrame['Kendalltau'])

            #buscamos la pocision del mayor y armamos un csv con la data de cada uno, ademas exportamos los modelos asociados
            indexR_Score = getIndexForMaxValues(dataFrame, maxR_Score, 'R_Score')
            indexPearson = getIndexForMaxValues(dataFrame, maxPearson, 'Pearson')
            indexSpearman = getIndexForMaxValues(dataFrame, maxSpearman, 'Spearman')
            indexKendalltau = getIndexForMaxValues(dataFrame, maxKendalltau, 'Kendalltau')

            indexList = [indexR_Score, indexPearson, indexSpearman, indexKendalltau]
            indexList = list(set(indexList))

            matrixSelected = []
            for index in indexList:

                #exportamos el modelo en formato joblib
                nameModel = pathResponse+"ModelExport"+str(index)+".joblib"
                dump(matrixModelData[index], nameModel)

                row = []
                for key in header:
                    row.append(dataFrame[key][index])
                matrixSelected.append(row)

            #exportamos los resumenes
            dataExport = pd.DataFrame(matrixSelected, columns=header)
            dataExport.to_csv(pathResponse+"summarySelectedModel.csv", index=False)

        except:
            print "Error during exec algorithm"
    else:
        print "Data set input not exist, please check the input for name file data set"
else:
    print "Path result not exist, please check input for path result"
