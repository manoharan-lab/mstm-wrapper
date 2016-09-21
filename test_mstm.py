# Copyright 2011-2013, 2016 Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Annie Stephenson, and
# Victoria Hwang
#
# This file is part of the mstm-wrapper project.
#
# This package is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this package.  If not, see <http://www.gnu.org/licenses/>.

"""
This package provides a python wrapper around the MSTM fortran-90 code
(http://www.eng.auburn.edu/~dmckwski/scatcodes/) written by Daniel Mackowski.
It produces input files, sends them to the mstm executable, parses the
output files, and calculates quantities of interest for static light scattering
measurements.  The mstm executable should be located in the user's path.

Based on work that was originally part of HoloPy
(https://github.com/manoharan-lab/holopy).

.. moduleauthor:: Anna Wang <annawang@seas.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import mstm
import numpy as np
from matplotlib import pyplot as plt

# make target object
#target = mstm.Target(np.array([1, 1]), np.array([1, 1]), np.array([0, 1]),
#                     np.array([0.125, 0.125]), 1.4, 1, 2)
target = mstm.Target(np.array([1]), np.array([1]), np.array([0]),
                     np.array([0.125]), 1.54, 1.0, 1)

# make incident object
incident = mstm.Incident((0, 1), [1, -1, 0, 0], np.array([15.7]))
    
# calculate the scattering matrix
scat_mat_data = mstm.calc_scat_matrix(target, incident, np.arange(0, 180, 1), np.array([0]))
    
# calculate the intensities
intensity_data = mstm.calc_intensity(target, incident, np.arange(0, 180, 1), np.array([0]))

plt.plot(np.arange(0, 180, 1),intensity_data[0,:,2])