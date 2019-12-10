########################################################################
# launcherClusteringScan.py,
#
# Script that allows to run the clustering service,
# input:
#     dataSet
#     Option normalize
#     pathResponse
# response:
#     csv error process
#     csv result process
#     histogram calinski
#     histogram silhouettes
#
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
# python launcherClusteringScan.py -d ../dataSetsDemo/vhl/dataCSV.csv -o 1 -p ../dataSetsDemo/vhl/result/ -r Clinical -k 1 -t 10

from modulesNLM.clustering_analysis import callService
from modulesNLM.utils import responseResults
from modulesNLM.clustering_analysis import evaluationClustering
from modulesNLM.utils import encodingFeatures

#import matplotlib
#matplotlib.use('Agg')
import graphviz as gp
import pylab
import sys
import pandas as pd
import argparse
import time
import glob

class Nodo(object):
    def __init__(self, data):
        # Data contiene el dataFrame de cada nodo
        self.data = data
        # Marca unica para realizar enlace y edges con graphviz
        self.id = int(round(time.time() * 1000))
        # rama izq
        self.left = None
        # rama der
        self.right = None

class BinaryTree(object):
    def __init__(self):
        self.top = None

    # Llamada para dividir grupo de forma recursiva
    def split(self, nodo, dataSet, pathResponse, optionNormalize, featureClass, kindDataSet, threshold, sizeEval):
        #print "Llamando a servicio -> ",dataSet.shape[0]
        callServiceObject = callService.serviceClustering(dataSet, pathResponse, optionNormalize, featureClass, kindDataSet, threshold, sizeEval)
        #callService debe retornar un arreglo en donde [sePuedeDividir,dataFramegrupo1,dataFramegrupo2]
        result = callServiceObject.execProcess()
        if isinstance(result,list):
            if(result[0] == -1):
                #print "No puedo dividir: ",dataSet.shape[0]
                dataSet.to_csv(pathResponse+""+str(dataSet.shape[0])+'_'+str(int(round(time.time() * 1000)))+'.csv', index=False)
                return nodo
            else:
                print "Dividir -> ",dataSet.shape[0]
                print "G1: ",result[1].shape[0]
                print "G2: ",result[2].shape[0]
                #Los sleep es para generar id unicos por cada dataframe que se agregaal arbol
                nodo.left = Nodo(result[1])
                time.sleep(0.05)
                nodo.right = Nodo(result[2])
                time.sleep(0.05)
                nodo.left = self.split(nodo.left,result[1], pathResponse, optionNormalize, featureClass, kindDataSet, threshold, sizeEval)
                time.sleep(0.05)
                nodo.right = self.split(nodo.right,result[2], pathResponse, optionNormalize, featureClass, kindDataSet, threshold, sizeEval)
                return nodo
        else:
            #almacena nodo anterior que se pudo dividir
            dataSet.to_csv(pathResponse+""+str(dataSet.shape[0])+'_'+str(int(round(time.time() * 1000)))+'.csv', index=False)
            return nodo

    # Llamar funcion recursiva para dibujar arbol
    def diagramSplit(self, pathResult) :
        print "Imprimir"
        tree = gp.Graph(format='png')
        if(self.top != None):
            self.draw(self.top,tree);
        # formatear pathResult quitando ultimo slash
        pathResult=pathResult[0:(len(pathResult)-1)]
        filename = tree.render(filename='tree',directory=pathResult)

    # Renderiza arbol de clustering
    def draw(self,data,tree):
        if(data.left != None):
            tree.edge(str(data.id),str(data.left.id));
            self.draw(data.left,tree)
        tree.node(str(data.id),str(data.data.shape[0]))
        if(data.right != None):
            tree.edge(str(data.id),str(data.right.id));
            self.draw(data.right,tree)

    # Insercion para el nodo raiz
    def insert(self, data):
        self.top = Nodo(data)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="full path and name to acces dataSet input process", required=True)
#parser.add_argument("-o", "--option", type=int, help="Option to Normalize data set: 1. Normal Scale\n2. Min Max Scaler\n3. Log scale\n4. Log normal scale", default=2)
parser.add_argument("-p", "--pathResult", help="full path for save results", required=True)
parser.add_argument("-r", "--response", help="Name response in dataset", required=True)
parser.add_argument("-k", "--kind", type=int, help="Kind of dataset: 1. Classification 2. Regression", required=True, choices=[1,2])
parser.add_argument("-t", "--threshold", type=float, help="threshold for umbalanced class", required=True)
parser.add_argument("-s", "--initialSize", type=int, help="initial Size of dataset", required=True)

args = parser.parse_args()

#hacemos las validaciones asociadas a si existe el directorio y el set de datos
processData = responseResults.responseProcess()#parser y checks...

if (processData.validatePath(args.pathResult) == 0):

    if (processData.validateDataSetExist(args.dataSet) == 0):

        #recibimos los datos de entrada...
        dataSet = pd.read_csv(args.dataSet)
        pathResponse = args.pathResult
        optionNormalize = 2
        featureClass = args.response
        kindDataSet = int(args.kind)
        threshold = float(args.threshold)
        #initialSize = int(args.initialSize)
        #Obtiene la cantida de row del dataSet
        initialSize = dataSet.shape[0];
        tree = BinaryTree()
        #Nodo raiz con la informacion del dataSet inicial
        tree.insert(dataSet)
        tree.split(tree.top,dataSet, pathResponse, optionNormalize, featureClass, kindDataSet, threshold, initialSize)
        tree.diagramSplit(pathResponse)

        #una vez procesada la data... para cada elemento en formato csv en el path response... lo leemos
        listFiles = glob.glob(pathResponse+"*.csv")

        matrixFull = []
        matrixGroup = []
        classResponse = []

        for files in listFiles:
            dataFrame = pd.read_csv(files)
            indexClass = files.split("/")[-1].split(".")[0]
            #armamos la matriz completa
            for i in range(len(dataFrame)):
                row = []
                for key in dataFrame:
                    row.append(dataFrame[key][i])#formamos la fila
                classResponse.append(indexClass)
                matrixGroup.append(row)

        #formamos el conjunto de datos para el entrenamiento del modelo de clasificacion
        dataFrameExport = pd.DataFrame(matrixGroup, columns=dataFrame.keys())
        del dataFrameExport[args.response]#eliminamos la columna respuesta

        #codificacion del conjunto de datos
        encoding = encodingFeatures.encodingFeatures(dataFrameExport, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        clusteringPerformance = evaluationClustering.evaluationClustering(dataSetNewFreq, classResponse)
        print "Calinski Performance: ", clusteringPerformance.calinski
        print "Silhouette score: ", clusteringPerformance.siluetas

        dataFrameExport['classGroup'] = classResponse
        dataFrameExport.to_csv(pathResponse+"fullDataSetWith.csv", index=False)
