import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="full path and name to acces dataSet input process", required=True)
parser.add_argument("-p", "--path", help="path to save dataset transform", required=True)
args = parser.parse_args()

dataSet = pd.read_csv(args.dataSet)
categoricalFeatures = []

#obtenemos los atributos del tipo categorico
for key in dataSet:

    for i in range(len(dataSet)):
        try:
            value = float(dataSet[key][i])
        except:
            categoricalFeatures.append(key)
            break

if len(categoricalFeatures)>0:
    print "Categorical features in dataset: ", categoricalFeatures

    dictEncoding = {}

    #comenzamos la transformacion por cada key
    for key in categoricalFeatures:

        #obtenemos los valores unicos
        uniqueValues = list(set(dataSet[key]))

        dictCategory = {}
        encodingData = 0#para trabajar con ordinal encoder
        for element in uniqueValues:

            dictCategory.update({element:encodingData})

            #modificamos los valores en el dataset
            for i in range(len(dataSet)):
                if dataSet[key][i] == element:
                    dataSet[key][i] = encodingData
            encodingData+=1

        #actualizamos el diccionario final
        dictEncoding.update({key:dictCategory})

    #exportamos el conjunto de datos
    dataSet.to_csv(args.path+"dataSetEncoding.csv", index=False)

    #exportamos el JSON con la data
    nameFileExport = "%sencodingProcessDict.json" % (args.path)
    with open(nameFileExport, 'w') as fp:
        json.dump(dictEncoding, fp)

else:
    print "There are not categorical features in dataset"
