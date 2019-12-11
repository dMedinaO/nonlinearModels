import argparse
import pandas as pd
from joblib import dump, load
import glob
import numpy as np

from modulesNLM.utils import encodingFeatures
#utils para el manejo de set de datos y su normalizacion
from modulesNLM.utils import transformDataClass
from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore

from modulesNLM.supervised_learning_predicction import performanceData
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, fbeta_score, make_scorer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modelPartition", help="Model created for to classify in a partition (*.joblib file)", required=True)
parser.add_argument("-d", "--dataset", help="Data set validation", required=True)
parser.add_argument("-p", "--pathModels", help="Path models of partitions", required=True)
parser.add_argument("-t", "--type", type=int, help="Type of model: 1. Class, 2. Regression", choices=[1,2], required=True)

args = parser.parse_args()

#primero, se hace la lectura del conjunto de datos recibido
dataset = pd.read_csv(args.dataset)

#estandarizar
applyNormal = ScaleNormalScore.applyNormalScale(dataset)
dataScaler = applyNormal.dataTransform

#cargamos el modelo de la clasificacion de grupos y lo usamos para predecir que usar
clfGroup = load(args.modelPartition)
classResponses = clfGroup.predict(dataScaler)#obtenemos las predicciones del grupo al que le corresponde

responsePredictedModel = []

for i in range(len(classResponses)):

    #obtenemos el ejemplo
    matrixExample = []#la cosa queda en una matriz para que sea correcta la clasificacion
    example = []
    for element in dataScaler:
        example.append(dataScaler[element][i])

    matrixExample.append(example)
    classValue = classResponses[i]#obtenemos la clasificacion

    #listamos todos los modelos existentes en el directorio asociado
    dirModel = args.pathModels+str(classValue)+"/"
    listModels = glob.glob(dirModel+"*.joblib")

    responsesForExample = []

    #cargamos los modelos y los usamos para predecir el valor
    for model in listModels:
        clfModel = load(model)
        response = clfModel.predict(matrixExample)
        responsesForExample.append(response[0])

    #obtenemos la prediccion final a partir de la combinacion de datos
    if args.type == 2:
        response = np.mean(responsesForExample)
        responsePredictedModel.append(response)
    else:
        #hacemos el sistema de votacion
        uniqueElement = list(set(responsesForExample))
        dictVotacion = {}
        listCont = []
        for element in uniqueElement:
            cont=0
            #contamos las veces que aparece
            for values in responsesForExample:
                if values == element:
                    cont+=1
            dictVotacion.update({str(element):cont})
            listCont.append(cont)

        maxCont = max(listCont)

        #buscamos el maximo en el diccionario y obtenemos la respuesta...
        for key in dictVotacion:
            if dictVotacion[key] == maxCont:
                responsePredictedModel.append(int(key))
                break

print "Predicted values: "
print responsePredictedModel
