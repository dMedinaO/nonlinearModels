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
from modulesNLM.statisticsCorroboration import checksUnityProcess, checkMembersInPartitions
from modulesNLM.utils import transformDataClass
from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore

from modulesNLM.utils import summaryScanProcess
from modulesNLM.utils import responseResults
from modulesNLM.utils import encodingFeatures
from modulesNLM.utils import processParamsDict

import json

dataSet = pd.read_csv(sys.argv[1])
pathOutput = sys.argv[2]
numberPartitions = int(sys.argv[3])
jsonResponse = sys.argv[4]

#codificamos el conjunto de datos para trabajar con elementos y poder evaluar las particiones
#procesamos el set de datos para obtener los atributos y las clases...
columnas=dataSet.columns.tolist()
x=columnas[len(columnas)-1]
targetResponse=dataSet[x]#clases
y=columnas[0:len(columnas)-1]
dataValues=dataSet[y]#atributos

encoding = encodingFeatures.encodingFeatures(dataValues, 20)
encoding.evaluEncoderKind()
dataSetNewFreq = encoding.dataSet

#ahora aplicamos el procesamiento segun lo expuesto
applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
data = applyNormal.dataTransform

#hacemos la lectura del JSON y evaluamos los elementos
siluetasScore = 0
calinskiScore = 0
with open(jsonResponse) as json_file:
    dataJSON = json.load(json_file)
    siluetasScore = float(dataJSON['silhouette_score'])
    calinskiScore = float(dataJSON['calinski_harabaz_score'])

checkNumberK = checksUnityProcess.checkRandomsPartitions(numberPartitions, pathOutput, data, calinskiScore, siluetasScore)
checkNumberK.checkRandomElement()

checkMembers = checkMembersInPartitions.checkMembersInPartitions(dataJSON['partitions_length'], pathOutput, data, calinskiScore, siluetasScore)
checkMembers.generateRandomPartitions()
