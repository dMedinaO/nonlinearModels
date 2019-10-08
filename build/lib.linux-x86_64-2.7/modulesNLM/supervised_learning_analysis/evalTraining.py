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

from sklearn.metrics import accuracy_score

class evalTraining(object):

    def __init__(self, classifiers, dataSetTesting, classTesting):
        self.classifiers = classifiers
        self.dataSetTesting = dataSetTesting
        self.classTesting = classTesting

        self.valuesPredic = self.classifiers.predict(self.dataSetTesting)#generamos los valores de prediccion

    #funcion que permite hacer la evaluacion del conjunto de datos
    def getPerformanceValues(self):

        print self.valuesPredic
        print self.classTesting
        accuracy = accuracy_score(self.classTesting, self.valuesPredic)
        print accuracy
