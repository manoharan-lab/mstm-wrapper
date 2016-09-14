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

from __future__ import division
import numpy as np
import subprocess
import tempfile
import glob
import os
import shutil


def separate_exponent(num):
    if num == 0:
        return 0,0
    else:
        return num/10**(np.floor(np.log10(np.abs(num)))), int(np.floor(np.log10(np.abs(num))))


def calc_scat_matrix(targ, inc, th_min, th_max, phi_min, phi_max, delete=True): # targ is target object, inc is incident object
    # print important things to temporary .inp file
    temp_dir = tempfile.mkdtemp()
    current_directory = os.getcwd()
    path, _ = os.path.split(os.path.abspath(__file__))
    #temp_dir = path # comment after debugging
    mstmlocation = os.path.join(path, 'mstm_ubuntu.exe')
    templatelocation = os.path.join(path, 'input_template.txt')
    shutil.copy(mstmlocation, temp_dir)
    shutil.copy(templatelocation, temp_dir)
    os.chdir(temp_dir)

    # make angles file
    th = []
    phi = []
    AngfileName = 'angles.dat'
    for i in range(th_min,th_max+1):
        for j in range(phi_min,phi_max+1):
            th.append(int(i))
            phi.append(int(j))
    angs = []
    for i in range(0,len(th)):
        angs.append(str(th[i])+ ',' +str(phi[i]))

    angfile = open(os.path.join(temp_dir, AngfileName),'w')
    for item in angs:
        angfile.write('%s\n' %item)
    angfile.close()

    OutputName = 'mstm_out.dat'
    length_scl_factor = 2*np.pi/inc.wavelength
    polarization_angle = np.arctan2(inc.jones_vec[1],inc.jones_vec[0])
    parameters = (targ.num_spheres, targ.index_spheres, targ.index_matrix, length_scl_factor, polarization_angle)
    g = ''
    for i in np.arange(0,targ.num_spheres):
        radb, rade = separate_exponent(targ.radii[i]) #have to make sure we don't print any e's into the text file.
        xb, xe = separate_exponent(targ.x[i]-np.mean(targ.x))
        print xb
        yb, ye = separate_exponent(targ.y[i]-np.mean(targ.y))
        zb, ze = separate_exponent(-(targ.z[i]-np.mean(targ.z)))
        g += '{0:20} {1:20} {2:20} {3:20}'.format(str(radb)+'d'+str(rade), str(xb)+'d'+str(xe), str(yb)+'d'+str(ye), str(zb)+'d'+str(ze))+'\n'
    with open(templatelocation, 'r') as InFile:
        InF = InFile.read()
    InF = InF.format(parameters, g, OutputName, AngfileName, len(angs) )
    InputFile = file(os.path.join(temp_dir, 'mstm.inp'), 'w')
    InputFile.write(InF)
    InputFile.close()

    # run MSTM
    cmd = ['./mstm_ubuntu.exe', 'mstm.inp']
    subprocess.check_call(cmd, cwd=temp_dir)

    # Go into results file
    result_file = glob.glob(os.path.join(temp_dir,'mstm_out.dat'))[0]
    print temp_dir
    # Read correct lines of results file
    mstm_result = iter(open(result_file))
    S11 = []
    S12 = []
    S13 = []
    S14 = []
    S11col = -1
    for line in mstm_result:
        print line
        l = line.split()
        if '11' in l:
            S11col = l.index('11')
            line = next(mstm_result)
            l = line.split()
        if len(l)>0 and S11col>-1:
            if 'matrix' in l:
                break
            if '************' in l[0]:
                break
            S11.append(float(l[S11col]))
            S12.append(float(l[S11col + 1]))
            S13.append(float(l[S11col + 2]))
            S14.append(float(l[S11col + 3]))
    Srow1 = [S11, S12, S13, S14]

    os.chdir(current_directory)
    if delete:
        shutil.rmtree(temp_dir)

    return Srow1, th, phi

def calc_Is(targ, inc, th_min, th_max, phi_min, phi_max):
    Srow1,th,phi = MSTM.calc_scat_matrix(self,targ, inc, th_min, th_max, phi_min, phi_max)
    prefactor = 1/((2*np.pi*targ.index_matrix/inc.wavelength)**2)
    #calculate Is using scattering matrix
    # still need to fix this
    Is = []#(Srow1[0]*inc.stokes_vec[0] + Srow1[1]*inc.stokes_vec[1] + Srow1[2]*inc.stokes_vec[2] + Srow1[3]*inc.stokes_vec[3])*prefactor*np.sin(th*np.pi/180)
    return Is

# create target class
class target:
    def __init__(self, x, y, z, radii, index_matrix, index_spheres, num_spheres):
        self.x = x
        self.y = y
        self.z = z
        self.radii = radii
        self.index_matrix = index_matrix
        self.index_spheres = index_spheres
        self.num_spheres = num_spheres

# create incident class
class incident:
    def __init__(self, jones_vec, stokes_vec, wavelength):
        self.jones_vec = jones_vec # jones vector
        self.stokes_vec = stokes_vec # stokes vector
        self.wavelength = wavelength # um


################
#### testing ##
##############
if __name__ == "__main__":
    t = target([1],[1],[0],[.125],1,1.4,1)
    inci = incident((1,0),(1,1,0,0),0.4)
    Sr,th, phi = calc_scat_matrix(t, inci, 0, 100, 0, 0)
    Is = calc_Is(t, inci, 0, 5, 0, 0)
