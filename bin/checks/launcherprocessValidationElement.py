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

from modulesNLM.utils import createRandomPartitions

import sys
import pandas as pd
import argparse
import os

#recibimos el conjunto de datos y el maximo de particiones, junto con el numero de veces a generar cada particion y el path
dataSet = pd.read_csv(sys.argv[1])
maxSplitter = int(sys.argv[2])
maxDistribution = int(sys.argv[3])
pathOutput = sys.argv[4]

for i in range(2, maxSplitter+1):#hacemos las iteraciones

    #creamos el directorio de la particion correspondiente
    command = "mkdir -p %s%d" % (pathOutput, i)
    os.system(command)

    for j in range(maxDistribution):
        randomDist = createRandomPartitions.createRandomDistribution(dataSet, i)#entregamos el set de datos y el numero de divisiones
        randomDist.createRandomValuesDistribution()
        break
    break
