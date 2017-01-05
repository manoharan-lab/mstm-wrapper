# Copyright 2011-2013, 2016 Vinothan N. Manoharan, Annie Stephenson, and
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
Tests for MSTM wrapper

.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import mstm
import numpy as np
import pandas as pd
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal


def test_single_sphere():
    """
    Test results for 1 sphere against structcol.mie and holopy.multisphere (F77)
    """

    # make target object
    xpos = [0]
    ypos = [0]
    zpos = [0]
    radii = [0.125]
    n_matrix = 1.54
    n_spheres = 1.33
    target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres)
    wavelength = 0.4, 0.5, 2
    theta = np.linspace(0, 180, 181)

    # calculate the cross section for random polarization
    calculation = mstm.MSTMCalculation(target, wavelength, theta, phi=None)
    result = calculation.run()
    total_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 0., 180.)
    refl_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 90., 180.)
    assert_almost_equal(total_csca[0], 0.011348999571838181, decimal = 6) # compare result with mie
    assert_almost_equal(refl_csca[0], 8.6652839744818315e-05, decimal = 7) # compare result with mie
    assert_almost_equal(total_csca[0], 0.011318622135323661, decimal = 4) # compare result with multisphere
    assert_almost_equal(refl_csca[0], 8.6407209410860873e-05, decimal = 6) # compare result with multisphere

def test_two_spheres():
    """
    Test results for 2 spheres against holopy.multisphere (F77)
    """

    # make target object
    xpos = [0,1]
    ypos = [0,0]
    zpos = [0,0]
    radii = [0.125, 0.125]
    n_matrix = 1.54
    n_spheres = 1.33
    target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres)
    wavelength = 0.4, 0.5, 2
    theta = np.linspace(0, 180, 181)

    # calculate the cross section for random polarization
    calculation = mstm.MSTMCalculation(target, wavelength, theta, phi=None)
    result = calculation.run()
    total_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 0., 180.)
    refl_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 90., 180.)
    assert_almost_equal(total_csca[0], 0.022626741536566294, decimal = 3) # compare result with multisphere
    assert_almost_equal(refl_csca[0], 0.0001636858558060008, decimal = 5) # compare result with multisphere

def test_three_spheres():
    """
    Test results for 3 spheres against holopy.multisphere (F77)
    """

    # make target object
    xpos = [0, 1, 0]
    ypos = [0, 0, 1]
    zpos = [0, 0, 0]
    radii = [0.125, 0.125, 0.125]
    n_matrix = 1.54
    n_spheres = 1.33
    target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres)
    wavelength = 0.4, 0.5, 2
    theta = np.linspace(0, 180, 181)

    # calculate the cross section for random polarization
    calculation = mstm.MSTMCalculation(target, wavelength, theta, phi=None)
    result = calculation.run()
    total_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 0., 180.)
    refl_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 90., 180.)
    assert_almost_equal(total_csca[0], 0.033904226206668324, decimal = 3) # compare result with multisphere
    assert_almost_equal(refl_csca[0], 0.00024189819424174622, decimal = 5) # compare result with multisphere

def test_random_orient_one_sphere():
    """
    Tests results for 1 sphere of random orientation against itself from a previous run
    """
     # make target object
    xpos = [0]
    ypos = [0]
    zpos = [0]
    radii = [0.125]
    n_matrix = 1.54
    n_spheres = 1.33
    target = mstm.Target(xpos, ypos, zpos, radii, n_matrix, n_spheres)
    wavelength = 0.4, 0.5, 2
    theta = np.linspace(0, 180, 181)

    # calculate the cross section for random polarization
    calculation = mstm.MSTMCalculation(target, wavelength, theta, phi = None, fixed = False)
    result = calculation.run()
    total_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 0., 180.)
    refl_csca = result.calc_cross_section(np.array([1, 0, 0, 0]), 90., 180.)
    assert_almost_equal(total_csca[0], 0.01134861, decimal = 5) # compare result with number from previous run
    assert_almost_equal(refl_csca[0], 8.66921190e-05, decimal = 5) # compare result with number from previous run

