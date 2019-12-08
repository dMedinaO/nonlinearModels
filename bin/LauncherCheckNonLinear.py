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

from modulesNLM.checks_module import checkNonLinearRegression, checkNonLinearClass
import sys
import pandas as pd
import argparse

#process data with argparse for to get a control of input params from command line
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help='dataSet input to check data', required=True)
parser.add_argument('--type', '-k', help='Kind of dataSet: 1 for classification models --- 2 for predictive models', type=int, choices=[1, 2], required=True)
parser.add_argument('--threshold', '-t', help='threshold to consider in linear evaluation of response in dataset', type=float, default=0.7)
parser.add_argument('--feature', '-f', help='Name of feature with response in dataset', required=True)
args = parser.parse_args()

#get inputs from command line
dataSet = args.dataset
tipo = args.type
featureResponse = args.feature
threshold = args.threshold

response = -1

if tipo == 1:#class
    checkMethod = checkNonLinearClass.checkNonLinearClass(dataSet, featureResponse, threshold)
    checkMethod.prepareDataSet()#preparamos el conjunto de datos
    response = checkMethod.applyTraining()#aplicamos la regresion lineal

else:
    checkMethod = checkNonLinearRegression.checkNonLinearRegression(dataSet, featureResponse, threshold)
    checkMethod.prepareDataSet()#preparamos el conjunto de datos
    response = checkMethod.applyLinearRegression()#aplicamos la regresion lineal

if response == 0:
    print "Dataset is nonlinear!\nPlease to use LauncherExploreModels.py script for to see more options of algorithms and parameters"
elif response == 1:
    print "Dataset is linear"
else:
    print "Error during exec process, please contac us at developer of library"
