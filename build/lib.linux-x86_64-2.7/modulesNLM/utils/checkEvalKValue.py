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

class checkKValue(object):

    def __init__(self, dataSet):

        self.dataSet = dataSet

        if len(self.dataSet)>=200:
            self.kvalue=10
        elif len(self.dataSet)<200 and len(self.dataSet)>=100:
            self.kvalue=5
        else:
            self.kvalue=-1#implica que se utilizara validacion cruzada con Leave One Out
