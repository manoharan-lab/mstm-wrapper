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

# set paramaters
num_spheres = 1
xpos = np.array([0])
ypos = np.array([0])
zpos = np.array([0])
radii = np.array([0.125])
n_matrix = 1.0
n_spheres = 1.54
theta = np.arange(90, 181, 1)
phi = np.arange(0, 361, 1)
jones_vec = [1, 0]
stokes_vec = [1, 1, 0, 0]
min_wavelength = 0.35
max_wavelength = 0.7
length_scl_factor = np.linspace(2*np.pi/max_wavelength, 2*np.pi/min_wavelength, num = 20)

# make the target object
target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres, num_spheres)
# make incident object
incident = mstm.Incident(jones_vec, stokes_vec, length_scl_factor)
# calculate the cross section
cross_section1 = mstm.calc_cross_section(target, incident, theta, phi)

# make incident object
incident2 = mstm.Incident([0, 1], [1, -1, 0, 0], length_scl_factor)
# calculate the cross section
cross_section2 = mstm.calc_cross_section(target, incident2, theta, phi)

plt.figure()
plt.plot(2*np.pi/length_scl_factor, cross_section1, 2*np.pi/length_scl_factor, cross_section2, 
         2*np.pi/length_scl_factor, (cross_section1+cross_section2)/2)
plt.legend(['1,0','0,1','averaged'])
plt.xlabel('Wavelength (um)')
plt.xlim([0.4,0.7])
plt.ylim([0,0.005])
plt.ylabel('Cross Section (um^2)')
plt.title('backscattering Cross Section')


