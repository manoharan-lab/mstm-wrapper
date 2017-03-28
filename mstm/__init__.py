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

import subprocess
import tempfile
import os
import shutil
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import integrate

class MSTMCalculation:
    """
    Input parameters and methods for running an MSTM calculation.

    Attributes
    ----------
    target : object of class Target
        coordinates and refractive indices of spheres
    wavelength : float or tuple
        wavelength at which to do calculation. If tuple, specifies start
        wavelength, end wavelength, and number of wavelengths.
    num_wavelengths : integer
        number of wavelengths
    theta : array
        polar angles (scattering angles)
    phi : array or None
        azimuthal angles. If phi=None, scattering matrix elements will be
        azimuthally averaged
    azimuthal_average : boolean
        True if azimuthal averaging is on
    fixed_orientation : boolean
        True if fixed orientation; False if random orientation. Not currently
        implemented

    Methods
    -------
    run()
        write input files, open connection to MSTM executable,
        and run the calculation
    """
    def __init__(self, target, wavelength, theta, phi=None, fixed=True,
                 mstm_executable = "mstm"):
        """
        Constructor for MSTMCalculation object.  Searches for the executable in
        the user's path and the package directory.

        Parameters
        ----------
        target : object of class Target
            coordinates and refractive indices of spheres
        wavelength : array
            wavelength (in vacuum) at which to do calculation
        theta : array
            polar angles (scattering angles)
        phi : array (optional)
            azimuthal angles. If phi=None, scattering matrix elements will be
            azimuthally averaged
        mstm_executable : string (optional)
            name of executable.
        """
        self.target = target
        self.wavelength = wavelength
        self.theta = theta
        self.phi = phi
        self.fixed = fixed
        if self.phi is None:
            self.azimuthal_average = True
        else:
            self.azimuthal_average = False
        # check if wavelength is tuple (has "len" attribute) or float
        if hasattr(self.wavelength, "__len__"):
            # if tuple, the third component specifies the number of wavelengths
            self.num_wavelengths = self.wavelength[2]
        else:
            self.num_wavelengths = 1

        # get path to current module
        module_dir, _ = os.path.split(os.path.abspath(__file__))

        # Search for the executable in the user's path
        # (relies on shutils.which(), which works only in Python 3.3 and above)
        mstm_path = shutil.which(mstm_executable)
        if mstm_path is None:
            # search in the path to the current module
            mstm_path = shutil.which(mstm_executable, path=module_dir)
            #mstm_path = module_dir + '/mstm.exe' # temporary fix for directory problems
        if mstm_path is None:
            raise RuntimeError("MSTM executable" + " \'" + mstm_executable +
                               "\' " + "not found")

        # underscore means "private" attribute
        self._mstm_path = mstm_path
        self._module_dir = module_dir

    def run(self, delete=True):
        """
        Run the calculation.

        Parameters
        ----------
        delete : boolean (optional)
            True if the temporary directory containing the generated input and
            output files should be deleted after running (default). Setting
            this to False might be useful in debugging.

        Returns
        -------
        MSTMresult object
            results of calculation
        """
        if self.num_wavelengths > 1:
            wavelengths = np.linspace(self.wavelength[0], self.wavelength[1], self.num_wavelengths)
        else:
            wavelengths = np.array([self.wavelength])

        # put input files in a temp directory
        temp_dir = tempfile.mkdtemp()
        current_dir = os.getcwd()
        os.chdir(temp_dir)

        # make angles file
        angle_filename = 'angles.dat'
        if self.azimuthal_average is True:
            phi = np.array([0.0])
        else:
            phi = self.phi
        thetatot = np.repeat(self.theta, len(phi))
        phitot = np.tile(phi, len(self.theta))
        angles = np.vstack((thetatot, phitot))
        angles = angles.transpose()

        for wl in wavelengths:
            wavevec = 2*np.pi/wl
            parameters = (self.target.num_spheres, self.target.index_spheres,
                          self.target.index_matrix)
            np.savetxt(os.path.join(temp_dir, angle_filename), angles, '%5.2f')

            # prepare input file for fortran code
            output_name = 'mstm_out1.dat'

            # format sphere sizes and positions
            radii = self.target.radii
            x = self.target.x
            y = self.target.y
            z = self.target.z
            sphere_str = ''
            for k in range(self.target.num_spheres):
                sphere_str += '{0:.10e} {1:.10e} {2:.10e} {3:.10e}\n'.\
                              format(radii[k], x[k], y[k], z[k])
            # convert to Fortran scientific notation, which uses 'd' instead of 'e'
            sphere_str = sphere_str.replace('e', 'd')

            # make string substitutions to the template and write to the input file
            template_path = os.path.join(self._module_dir, 'input_template.txt')
            with open(template_path, 'r') as template_file:
                template = template_file.read()
            mstm_input = template.format(target = parameters,
                                         spheres = sphere_str,
                                         output_file = output_name,
                                         azimuth_average = \
                                         int(self.azimuthal_average),
                                         random_orientation = int(not self.fixed),
                                         scattering_angle_file = angle_filename,
                                         number_scattering_angles = len(angles),
                                         length_scale_factor = wavevec )
            input_file = open(os.path.join(temp_dir, 'mstm.inp'), 'w')
            input_file.write(mstm_input)
            input_file.close()

            # run MSTM fortran executable
            cmd = [self._mstm_path, 'mstm.inp']
            subprocess.check_call(cmd, cwd=temp_dir)
            with open(os.path.join(temp_dir, output_name)) as f1:
                with open(os.path.join(temp_dir, 'mstm_out.dat'), 'a') as f:
                    for line in f1:
                            f.write(line)

        # Read from results file
        result = MSTMResult(os.path.join(temp_dir, 'mstm_out.dat'), self)

        # delete temp files
        os.chdir(current_dir)
        if delete:
            shutil.rmtree(temp_dir)

        return result



class MSTMResult:
    """
    Results from running MSTM calculation.  Includes values of the extinction,
    absorption, and scattering efficiencies, and scattering matrix elements as
    a function of theta (and phi, if no azimuthal averaging has been done).

    Attributes
    ----------
    scattering_matrix : pandas DataFrame
        columns are theta, (phi), values of S11-S44
        rows are different angles
    efficiencies : pandas DataFrame
        columns: qext, qabs, qsca
        rows: unpolarized, parallel, perpendicular
    asymmetry : float
        asymmetry parameter
    mstm_calculation : object of type MSTMCalculation
        input values used to do the calculation

    Methods
    -------
    calc_intensity(stokes)
        calculate scattered intensity for given Stokes vector.
    calc_cross_section(stokes, theta_min, theta_max)
        calculate cross-section by integrating azimuthally-averaged intensities
    """
    def __init__(self, output_filename, mstm_calculation):
        """
        Constructor for MSTMResult object.

        Parameters
        ----------
        output_filename : string
            path to MSTM output file
        mstm_calculation : object of type MSTMCalculation
            stores input values used to do the calculation
        """
        self.mstm_calculation = mstm_calculation
        if mstm_calculation.phi is not None:
            num_angles = len(mstm_calculation.theta)*len(mstm_calculation.phi)
        else:
            num_angles = len(mstm_calculation.theta)

        with open(output_filename, "r") as resultfile:
            mstm_result = resultfile.readlines()
        scat_mat_headers = [i for i, j in enumerate(mstm_result)
                            if j.startswith(' scattering matrix elements')]
        if mstm_calculation.fixed == False:
            qsca_headers =  [i for i, j in enumerate(mstm_result)
                        if j.startswith(' total ext, abs, scat efficiencies')]
        else:
            qsca_headers = [i for i, j in enumerate(mstm_result)
                        if j.startswith(' unpolarized total ext')]

        self.scattering_matrix = []
        for row in scat_mat_headers:
            # need to disable "skip_blank_lines" or the row numbers of the
            # headers won't match those in the file
            if mstm_calculation.fixed == True:
                nr = num_angles
            if mstm_calculation.fixed == False:
                nr = num_angles + 1 # random orientation calculations add a blank line after headers
            dataframe = pd.read_table(output_filename, header = row + 1,
                                      nrows = nr,
                                      delim_whitespace = True,
                                      skip_blank_lines = False)

            dataframe = dataframe[pd.notnull(dataframe['14'])]
            dataframe = dataframe.astype(float)

            self.scattering_matrix.append(dataframe)

        self.efficiencies = []
        self.asymmetry = []
        for row in qsca_headers:
            if mstm_calculation.fixed == True:
                dataframe = pd.DataFrame(columns=['qext', 'qabs', 'qsca'],
                                         index=['unpolarized', 'par', 'perp'])
                # read unpolarized values first; this line also contains g
                [qext, qabs, qsca, g] = [float(num) for num in
                                         mstm_result[row+1].split()]
                dataframe.loc['unpolarized'] = [qsca, qabs, qsca]
                dataframe.loc['par'] = [float(num) for num in
                                        mstm_result[row+3].split()]
                dataframe.loc['perp'] = [float(num) for num in
                                         mstm_result[row+5].split()]
            if mstm_calculation.fixed == False:
                dataframe = pd.DataFrame(columns=['qext', 'qabs', 'qsca'],
                                         index=['unpolarized'])
                # read unpolarized values first; this line also contains g
                [qext, qabs, qsca, g] = [float(num) for num in
                                         mstm_result[row+1].split()]
                dataframe.loc['unpolarized'] = [qsca, qabs, qsca]
            self.efficiencies.append(dataframe)
            self.asymmetry.append(g)

        if self.mstm_calculation.num_wavelengths > 1:
            self.wavelength = np.linspace(self.mstm_calculation.wavelength[0],
                                          self.mstm_calculation.wavelength[1],
                                          self.mstm_calculation.num_wavelengths)
        else:
            self.wavelength = np.array([self.mstm_calculation.wavelength])


    def calc_intensity(self, stokes):
        """
        Calculate intensity, given the incident Stokes vector

        Parameters
        ----------
        stokes : array
            incident Stokes vector (one-dimensional, 4 elements: I, Q, U, V)

        Returns
        -------
        list of DataFrames :
            one for each wavelength; each DataFrame contains three columns:
            theta, phi, intensity (two columnes when azimuthally averaged)
        """

        intensity = []
        # calculate volume mean radius and geometrical cross-section
        vm_radius = self.mstm_calculation.target.volmean_radius()
        geometric_cross_sec = np.pi*vm_radius**2

        for i in range(self.mstm_calculation.num_wavelengths):
            # S11 (the so-called "phase function") is normalized so that 1/4pi
            # times its integral over all angles is equal to 1.  So we have to
            # correct by multiplying it by a factor of Csca (see Bohren and
            # Huffman sec 3.4 page 73) to get the differential scattering
            # cross-section.
            qsca = self.efficiencies[i].loc['unpolarized', 'qsca']
            csca = qsca*geometric_cross_sec
            # we need to divide by the matrix refractive index for the results
            # to agree with Mie theory for a single spheres. It's not clear why
            # this factor has to appear here.
            prefactor = csca/(2.0*self.mstm_calculation.target.index_matrix)

            mat = self.scattering_matrix[i]
            intensities = prefactor*(mat['11']*stokes[0] +
                                     mat['12']*stokes[1] +
                                     mat['13']*stokes[2] +
                                     mat['14']*stokes[3])
            # label the column in the dataframe
            intensities.name = 'intensity'

            if (self.mstm_calculation.azimuthal_average is True or
                self.mstm_calculation.fixed is False):
                dataframe = pd.concat([mat['theta'], intensities], axis = 1)
            else:
                dataframe = pd.concat([mat['theta'], mat['phi'], intensities],
                                      axis = 1)
            intensity.append(dataframe)

        return intensity

    def calc_cross_section(self, stokes, theta_min, theta_max):
        """
        Calculate the cross section from wavelength.
        If, for example, theta = 0-180, the cross section calculated is the
        total cross section. If theta = 90-180, the cross section calculated is
        the reflection cross section, which is proportional to the
        reflectivity.

        Parameters
        ----------
        target : an object of the Target class
        theta_min : float
            lower integration limit on theta, in degrees
        theta_max : float
            upper integration limit on theta, in degrees

        Returns
        -------
        numpy array:
        cross_section
        """
        # TODO : should check to make sure theta_min and theta_max are within
        # the range of theta that has been calculated
#        if self.mstm_calculation.azimuthal_average is False:
#            raise ValueError("This calculation requires azimuthal averaging;\n"+
#                             "Make sure phi=None in the MSTM calculation.")
        num_wavelengths = self.mstm_calculation.num_wavelengths
        intensity = self.calc_intensity(stokes)
        cross_section = np.zeros(num_wavelengths)
        for i in range(num_wavelengths):
            theta = intensity[i]['theta'].as_matrix()
            theta = theta * np.pi/180.
            integrand = (intensity[i]['intensity']*np.sin(theta))
            f = interpolate.interp1d(theta, integrand.as_matrix())
            cross_section[i], err = integrate.quad(f, theta_min*np.pi/180,
                                                   theta_max*np.pi/180)

        return cross_section

    def calc_reflectance(self, stokes):
        """
        Calculation of the reflectance as a function of wavelength.
        Valid only for assemblies where the volume mean radius is greater than 
        the incident wavelength

        Parameters
        ----------
        stokes: stokes vector of incident light

        Returns
        -------
        numpy array:
        R

        Notes
        -----
        See page 122 of Bohren-Huffman for derivation of formula

        """
        vm_radius = self.mstm_calculation.target.volmean_radius()
        R = self.calc_cross_section(stokes, 90, 180)/(np.pi*vm_radius**2)
        return R

class Target:
    """
    Class to contain data describing the sphere assemblies that scatter the
    light

    Attributes
    ----------
    x : array
        x-coordinates of spheres in assembly
    y : array
        y-coordinates of spheres in assembly
    z : array
        z-coordinates of spheres in assembly
    radii : array
        radii of spheres in assembly
    index_matrix : array
        refractive index of medium surrounding spheres
    index_spheres : array
        refractive index of spheres
    assembly_radius : float
        radius of sphere encompassing the assembly of spheres. Default set to 
        None for asssemblies 
        

    Notes
    -----
    x, y, z, radii, and assembly_radius must be in same units, and must 
    also match units of wavelength of incident light
    """
    def __init__(self, x, y, z, radii, index_matrix, index_spheres, 
                 assembly_radius = None):
        """
        Constructor for object of the Target class.
        """
        x = x-np.mean(x)
        y = y-np.mean(y)
        z = z-np.mean(z)
        self.index_matrix = index_matrix
        self.index_spheres = index_spheres
        self.assembly_radius = assembly_radius
    
        if self.assembly_radius != None:
            index = np.where(x**2 + y**2 + z**2 < self.assembly_radius**2)
            self.x = x[index]
            self.y = y[index]
            self.z = z[index]
            self.radii = radii[index]

        else:
            self.x = x
            self.y = y
            self.z = z
            self.radii = radii
            
        self.num_spheres = len(self.x)

    def volmean_radius(self):
        """
        Calculates volume-mean radius.  This quantity is needed to calculate
        the (dimensional) cross-sections from the efficiencies.
        """
        return np.power(np.sum(np.power(self.radii, 3)), 1./3)
