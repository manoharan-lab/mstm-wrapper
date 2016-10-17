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
import pandas as pd
import matplotlib.pyplot as plt
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal

#############################################################################
## Test results for 1 sphere against mie and F77 holopy code (multisphere) ##
#############################################################################

# make target object
xpos = np.array([0])
ypos = np.array([0])
zpos = np.array([0])
radii = np.array([0.125])
n_matrix = 1.0
n_spheres = 1.54
target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres)
wavelength = 0.4, 0.7, 20
theta = np.linspace(0, 180, 181)

# calculate the cross section for random polarization
calculation = mstm.MSTMCalculation(target, wavelength, theta, phi=None)
result = calculation.run()
total_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 0., 180.)
back_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 90., 180.)
assert_almost_equal(total_csca[0], 9.934121133041029506e-02, decimal = 5) # compare result with mie

#plt.figure()
#plt.plot(result.wavelength, (total_csca), linewidth = 1.0)
#plt.legend(['unpolarized (from MSTM F90)'])
#plt.xlabel('Wavelength (um)')
#plt.xlim([0.4,0.7])
#plt.ylim([0.0, 0.10])
#plt.ylabel('Cross Section (um^2)')
#plt.title('total Cross Section')
