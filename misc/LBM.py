# LatticeBoltzmannDemo.py:  a two-dimensional lattice-Boltzmann "wind tunnel" simulation
# Uses numpy to speed up all array handling.
# Uses matplotlib to plot and animate the curl of the macroscopic velocity field.

# Copyright 2013, Daniel V. Schroeder (Weber State University) 2013

# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated data and documentation (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

# Except as contained in this notice, the name of the author shall not be used in 
# advertising or otherwise to promote the sale, use or other dealings in this 
# Software without prior written authorization.

# Credits:
# The "wind tunnel" entry/exit conditions are inspired by Graham Pullan's code
# (http://www.many-core.group.cam.ac.uk/projects/LBdemo.shtml).  Additional inspiration from 
# Thomas Pohl's applet (http://thomas-pohl.info/work/lba.html).  Other portions of code are based 
# on Wagner (http://www.ndsu.edu/physics/people/faculty/wagner/lattice_boltzmann_codes/) and
# Gonsalves (http://www.physics.buffalo.edu/phy411-506-2004/index.html; code adapted from Succi,
# http://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199679249).

# For related materials see http://physics.weber.edu/schroeder/fluids

import numpy

# Define constants:
height = 80							# lattice dimensions
width = 200
viscosity = 0.02					# fluid viscosity
omega = 1 / (3*viscosity + 0.5)		# "relaxation" parameter
u0 = 0.1							# initial and in-flow speed
four9ths = 4.0/9.0					# abbreviations for lattice-Boltzmann weight factors
one9th   = 1.0/9.0
one36th  = 1.0/36.0
performanceData = False				# set to True if performance data is desire
# Initialize barriers:
barrier = numpy.zeros((height,width), bool)					# True wherever there's a barrier
barrier[int(height/2)-8:int(height/2)+8, int(height/2)] = True			# simple linear barrier
barrierN = numpy.roll(barrier,  1, axis=0)					# sites just north of barriers
barrierS = numpy.roll(barrier, -1, axis=0)					# sites just south of barriers
barrierE = numpy.roll(barrier,  1, axis=1)					# etc.
barrierW = numpy.roll(barrier, -1, axis=1)
barrierNE = numpy.roll(barrierN,  1, axis=1)
barrierNW = numpy.roll(barrierN, -1, axis=1)
barrierSE = numpy.roll(barrierS,  1, axis=1)
barrierSW = numpy.roll(barrierS, -1, axis=1)




class LBMSimulator:
	def __init__(self) -> None:		
		# Initialize all the arrays to steady rightward flow:
		self.n0 = four9ths * (numpy.ones((height,width)) - 1.5*u0**2)	# particle densities along 9 directions
		self.nN = one9th * (numpy.ones((height,width)) - 1.5*u0**2)
		self.nS = one9th * (numpy.ones((height,width)) - 1.5*u0**2)
		self.nE = one9th * (numpy.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nW = one9th * (numpy.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nNE = one36th * (numpy.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nSE = one36th * (numpy.ones((height,width)) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nNW = one36th * (numpy.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nSW = one36th * (numpy.ones((height,width)) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.rho = self.n0 + self.nN + self.nS + self.nE + self.nW + self.nNE + self.nSE + self.nNW + self.nSW		# macroscopic density
		self.ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / self.rho				# macroscopic x velocity
		self.uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / self.rho				# macroscopic y velocity



	 # Move all particles by one step along their directions of motion (pbc):
	def stream(self):
		self.nN  = numpy.roll(self.nN,   1, axis=0)	# axis 0 is north-south; + direction is north
		self.nNE = numpy.roll(self.nNE,  1, axis=0)
		self.nNW = numpy.roll(self.nNW,  1, axis=0)
		self.nS  = numpy.roll(self.nS,  -1, axis=0)
		self.nSE = numpy.roll(self.nSE, -1, axis=0)
		self.nSW = numpy.roll(self.nSW, -1, axis=0)
		self.nE  = numpy.roll(self.nE,   1, axis=1)	# axis 1 is east-west; + direction is east
		self.nNE = numpy.roll(self.nNE,  1, axis=1)
		self.nSE = numpy.roll(self.nSE,  1, axis=1)
		self.nW  = numpy.roll(self.nW,  -1, axis=1)
		self.nNW = numpy.roll(self.nNW, -1, axis=1)
		self.nSW = numpy.roll(self.nSW, -1, axis=1)
		# Use tricky boolean arrays to handle barrier collisions (bounce-back):
		self.nN[barrierN] = self.nS[barrier]
		self.nS[barrierS] = self.nN[barrier]
		self.nE[barrierE] = self.nW[barrier]
		self.nW[barrierW] = self.nE[barrier]
		self.nNE[barrierNE] =self. nSW[barrier]
		self.nNW[barrierNW] =self. nSE[barrier]
		self.nSE[barrierSE] =self. nNW[barrier]
		self.nSW[barrierSW] =self. nNE[barrier]


	# Collide particles within each cell to redistribute velocities (could be optimized a little more):
	def collide(self):
	
		rho = self.n0 + self.nN + self.nS + self.nE + self.nW + self.nNE + self.nSE + self.nNW + self.nSW
		ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / rho
		uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / rho
		self.ux ,self.uy= ux, uy
		ux2 = ux * ux				# pre-compute terms used repeatedly...
		uy2 = uy * uy
		u2 = ux2 + uy2
		omu215 = 1 - 1.5*u2			# "one minus u2 times 1.5"
		uxuy = ux * uy
		self.n0 = (1-omega)*self.n0 + omega * four9ths * rho * omu215
		self.nN = (1-omega)*self.nN + omega * one9th * rho * (omu215 + 3*uy + 4.5*uy2)
		self.nS = (1-omega)*self.nS + omega * one9th * rho * (omu215 - 3*uy + 4.5*uy2)
		self.nE = (1-omega)*self.nE + omega * one9th * rho * (omu215 + 3*ux + 4.5*ux2)
		self.nW = (1-omega)*self.nW + omega * one9th * rho * (omu215 - 3*ux + 4.5*ux2)
		self.nNE = (1-omega)*self.nNE + omega * one36th * rho * (omu215 + 3*(ux+uy) + 4.5*(u2+2*uxuy))
		self.nNW = (1-omega)*self.nNW + omega * one36th * rho * (omu215 + 3*(-ux+uy) + 4.5*(u2-2*uxuy))
		self.nSE = (1-omega)*self.nSE + omega * one36th * rho * (omu215 + 3*(ux-uy) + 4.5*(u2-2*uxuy))
		self.nSW = (1-omega)*self.nSW + omega * one36th * rho * (omu215 + 3*(-ux-uy) + 4.5*(u2+2*uxuy))
		# Force steady rightward flow at ends (no need to set 0, N, and S components):
		self.nE[:,0] = one9th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nW[:,0] = one9th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nNE[:,0] = one36th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nSE[:,0] = one36th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nNW[:,0] = one36th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
		self.nSW[:,0] = one36th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
	


	



# Compute curl of the macroscopic velocity field:
def curl(ux, uy):
	return numpy.roll(uy,-1,axis=1) - numpy.roll(uy,1,axis=1) - numpy.roll(ux,-1,axis=0) + numpy.roll(ux,1,axis=0)

def velocity_field(ux, uy):
    return ux, uy


def matplotScalarFieldVisualize(simulator, scalarFieldFunc):
	import time, matplotlib.pyplot, matplotlib.animation
	# Here comes the graphics and animation...
	theFig = matplotlib.pyplot.figure(figsize=(8,3))
	# ax = theFig.add_subplot(111)
	fluidImage = matplotlib.pyplot.imshow(scalarFieldFunc(simulator.ux, simulator.uy), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1), 
										cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')

	# x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))
	# quiverImage = ax.quiver(x, y, ux, uy, angles='xy', scale_units='xy', scale=0.1, color='black')


			
	bImageArray = numpy.zeros((height, width, 4), numpy.uint8)	# an RGBA image
	bImageArray[barrier,3] = 255								# set alpha=255 only at barrier sites
	barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')

	# Function called for each successive animation frame:
	start_time = time.perf_counter()
	#frameList = open('frameList.txt','w')		# file containing list of images (to make movie)
	def nextFrame(arg):							# (arg is the frame number, which we don't need)
		global startTime
		if performanceData and (arg%100 == 0) and (arg > 0):
			end_time = time.perf_counter()
			print(f"Elapsed time: {end_time - start_time} seconds")
			startTime = end_time
		#frameName = "frame%04d.png" % arg
		#matplotlib.pyplot.savefig(frameName)
		#frameList.write(frameName + '\n')
		for step in range(20):					# adjust number of steps for smooth animation
			simulator.stream()
			simulator.collide()
		# u, v = velocity_field(ux, uy)
		fluidImage.set_array(scalarFieldFunc(simulator.ux, simulator.uy))
		return (fluidImage, barrierImage)		# return the figure elements to redraw
		# quiverImage.set_UVC(u, v)
		# return (quiverImage, barrierImage)
	

	animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=True)
	matplotlib.pyplot.show()

def matplotVectorFieldVisualize(simulator, vectorFieldFunc):
	import time, matplotlib.pyplot, matplotlib.animation
	# Here comes the graphics and animation...
	theFig = matplotlib.pyplot.figure(figsize=(8,3))
	ax = theFig.add_subplot(111)
	
	x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))
	quiverImage = ax.quiver(x, y,simulator.ux, simulator.uy, angles='xy', scale_units='xy', scale=0.1, color='black')


			
	bImageArray = numpy.zeros((height, width, 4), numpy.uint8)	# an RGBA image
	bImageArray[barrier,3] = 255								# set alpha=255 only at barrier sites
	barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')

	# Function called for each successive animation frame:
	start_time = time.perf_counter()
	#frameList = open('frameList.txt','w')		# file containing list of images (to make movie)
	def nextFrame(arg):							# (arg is the frame number, which we don't need)
		global startTime
		if performanceData and (arg%100 == 0) and (arg > 0):
			end_time = time.perf_counter()
			print(f"Elapsed time: {end_time - start_time} seconds")
			startTime = end_time
		#frameName = "frame%04d.png" % arg
		#matplotlib.pyplot.savefig(frameName)
		#frameList.write(frameName + '\n')
		for step in range(20):					# adjust number of steps for smooth animation
			simulator.stream()
			simulator.collide()
		u, v = vectorFieldFunc(simulator.ux, simulator.uy)
		quiverImage.set_UVC(u, v)
		return (quiverImage, barrierImage)
	

	animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=True)
	matplotlib.pyplot.show()

if __name__ == '__main__':
	sim=LBMSimulator() 

	matplotScalarFieldVisualize(sim,curl)
	# matplotVectorFieldVisualize(sim, velocity_field)