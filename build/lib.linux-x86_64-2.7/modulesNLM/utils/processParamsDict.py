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

import pandas as pd

class processParams(object):

    def __init__(self, pathExport, listPerformance):

        self.pathExport = pathExport
        self.listPerformance = listPerformance

    #funcion que permite formar un diccionario con respecto a los parametros dependiendo del tipo de algoritmo
    def getDictParamsData(self, paramsValues):

        dictResponse = {}

        listParams = paramsValues.split("-")
        for paramsData in listParams:
            data = paramsData.split(":")
            try:
                dictResponse.update({data[0]:data[1]})
            except:
                dictResponse.update({"paramsvalue": "Default"})
        return dictResponse

    #funcion que permite obtener el mejor modelo en cada medida
    def getBestModels(self):

        self.listModels = []

        for measure in self.listPerformance:

            dictMeasure = {}
            dataModels = pd.read_csv(self.pathExport+measure+"_ranking.csv")
            algorithm = dataModels['Algorithm'][0]
            params = dataModels['Params'][0]
            dictMeasure.update({"algorithm":algorithm})

            for measureV in self.listPerformance:

                dictMeasure.update({measureV:dataModels[measureV][0]})

            #agregamos la informacion del diccionario de los parametros
            dictMeasure.update({"params":self.getDictParamsData(params)})
            self.listModels.append(dictMeasure)
