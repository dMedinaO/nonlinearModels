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

import numpy.linalg

def findE(x,b): # Function to find the matrix "E"
    x_inv = numpy.linalg.pinv(x) # Find the Moore-Penrose Inverse of x
    W = numpy.dot(x_inv, b) # Multiple x# and b
    E = numpy.subtract(numpy.dot(x,W), b) # Subtract b from x times W
    E_rounded = numpy.around(E.astype(numpy.double), 1) # Round the E matrix to one decimal point
    return E_rounded, W

def checkEEZ(E): # Check if the matrix "E" is equal to zero
    for row in E:
        if (row != 0):
            return False
    return True

def checkELZ(E): # Check if the matrix "E" is less than zero
    for row in E:
        if (row > 0 or row == 0):
            return False
    return True

def done(W): #Print out the equation of the line
   print "Equation of the line is x =", ((W[2]*-1)/W[0]) # Print statement for 2D problems
   #print "Equation of the line is x - y + z =", ((W[3]*-1)/W[0]) # Print statement for 3D problem

x = [[0,0,1], [0,1,1], [-1,0,-1], [-1,-1,-1]] # Example one, solution is possible
#x = [[0,0,1], [1,1,1], [0,-1,-1], [-1,0,-1]] # Example two, solution is not possible
#x = [[0,0,1,1], [1,0,0,1], [1,0,1,1], [1,1,1,1],
#     [0,0,0,-1], [0,-1,0,-1], [0,-1,-1,-1], [-1,-1,0,-1]] # Example three, 3D problem, solution is a plane
b = [1,0,1,0]
while (True): # Keep running until we find a solution or a solution is not possible
    E,W = findE(x,b)
    EEZ = checkEEZ(E)
    ELZ = checkELZ(E)

    if (EEZ):
        done(W)
        break
    elif (ELZ):
        print "No solution is possible"
        break
b = numpy.add(b, numpy.add(E, numpy.absolute(E))) # Add b to the addition of E and the absolute value of E
