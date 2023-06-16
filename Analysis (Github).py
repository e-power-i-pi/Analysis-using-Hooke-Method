import numpy as np
import warnings as w
from numpy import sin, cos
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(567)

class Find_decent_Vrel:

	def __init__(self, pos, vel, dist):

		statement = "\nOrder of parameters: [X, Y, Z]; [Vx, Vy, Vz]; distance"
		w.warn(statement, SyntaxWarning, stacklevel = 2)

		self.x, self.y, self.z = pos
		self.vx, self.vy, self.vz = vel
		self.dist = dist

	# ------------------------------- #

	def check_Vrel(self, minrange, maxrange):

		bool_list = []
		temp = np.array((1/self.dist) * ( (self.x * self.vx) + (self.y * self.vy) + (self.z * self.vz) ))
		for i in temp:
			if (minrange < i < maxrange):
				bool_list.append(True)
			else:
				bool_list.append(False)

		return bool_list



# ------------------------------- # ------------------------------- #



class Analysis_using_Hooke:

	# set_rotation = False
	# dGC = 8 # kpc (distance of GC from earth)

	gayy = 1e-13 # GeV^-1
	r0 = 10 # km
	mass_NS = 1 # Msun
	rho_inf = 6.5 * 1e4 # 
	vDM = 200 # km/s (Dark Matter Velocity) 
	f_the_factor = ( (1.6 * 1e-19) / (6.62607015 * 1e-34) ) * 1e-9
	ma = 1e-5 * f_the_factor  # in GHz. ma = 10 micro eV/c^2
	c = 299792.458 # km/s

	beamFWHM = 0.124159 * 1e-2 # GBT

	def __init__(self, B, P, alpha, pos, vel, dist, time = 0):

		statement1 = "Order of parameters: B (in G), P (in seconds), alpha (in radiens),"
		statement2 = " co-ordinates(x, y, z) (in kpc), velocity(vx, vy, vz) (in km/s),"
		statement3 = " distance (in kpc) and time (in Million years)."
		w.warn(statement1 + statement2 + statement3, SyntaxWarning, stacklevel = 2)

		self.B = np.array(B) # B0 in G
		self.P = np.array(P) # P in s
		self.alpha = np.array(alpha) 
		self.x, self.y, self.z = pos
		self.vx, self.vy, self.vz = vel
		self.dist = dist
		self.time = time
		self.arrind = np.linspace(0, B.size - 1, B.size, dtype = int)

		if((self.B).size != (self.P).size or (self.P).size != (self.alpha).size):
			print((self.B).size, (self.P).size, (self.alpha).size)
			raise ValueError("Array sizes of B, P and alpha are not same.")

	# ------------------------------- #

	def power_hooke(self):

		# formula

		f_the_factor = ( (1.6 * 1e-19) / (6.62607015 * 1e-34) ) * 1e-9
		ma = self.ma # in GHz. ma = 10 micro eV/c^2
		Omega = 2 * np.pi / self.P
		theta = np.random.random(size = (self.B).size) * (np.pi/2)

		m_dot_r = (cos(self.alpha) * cos(theta)) + (sin(self.alpha) * sin(theta) * cos(Omega * self.time))

				# Will remain constant
		part1 = 4.5 * 1e8 * ((self.gayy / 1e-12)**2) * ((self.r0/10)**2) * ((ma/1)**(5/3)) 
				# Will remain constant but can change later
		part2 = (self.mass_NS/1) * (200/self.vDM) * (self.rho_inf/0.3)
				# Is an array of values 
		part3 = ((self.B/1e14)**(2/3)) * ((self.P/1)**(4/3))
		part4 = ( (3 * m_dot_r * m_dot_r) + 1 ) / (np.abs( (3 * cos(theta) * m_dot_r) - cos(self.alpha) )**(4/3))

		return part1 * part2 * part3 * part4

	# ------------------------------- #

	def distance_from_earth(self):
		x,y,z = self.x, self.y, self.z

		distance = np.sqrt( ((x)**2) + ((y)**2) + ((z)**2) )
		return distance

	# ------------------------------- #

	def referenced_sorting(self, X, Y):
		Y_sort = np.array([x for _, x in sorted(zip(X, Y))]) # X dictates order
		return Y_sort

	# ------------------------------- #

	def fobs(self):
		d = self.dist
		Vp = np.array([self.vx, self.vy, self.vz])
			# vx, vy, vz are from GC. As of now earth is stationary so Vp doesn't change.

		n_cap = (1/d) * (np.array([self.x, self.y - 8.5, self.z]))
			# n_cap is a unit vector from the sun (0, 8.5 ,0) to the pulsar

		newf = np.array([])
		# for i in range(d.size):
		# 	Vrel = np.dot(Vp[:, i], n_cap[:, i])
		# 	newf = np.append(newf, self.ma * (1 + ( Vrel/self.c )))

		Vrel = ( Vp[0] * n_cap[0] ) + ( Vp[1] * n_cap[1] ) + ( Vp[2] * n_cap[2] )
		newf = self.ma * (1 + ( Vrel/self.c ))
		print(Vrel.size, newf.size)

		return newf

	# ------------------------------- #

	def allsky_flux_density(self, doppf):
		L = self.power_hooke()
		d = self.dist
		delf = np.arange(np.min(doppf), np.max(doppf), 1e-4)
		if(delf[-1] != np.max(doppf)):
			delf = np.append(delf, np.max(doppf))

		if(delf.size < 1):
			raise ValueError("First find doppler shifted frequency, then use this function.")


		index = self.referenced_sorting(doppf, self.arrind)
		doppf_reord = np.sort(doppf)

		S = np.array([])
		num = np.array([], dtype = int)

		j, temp1, n  = 0, 0, 0
		for i in range(delf.size - 1):
			while(doppf_reord[j] <= delf[i+1]):
				temp1 = temp1 + (L[index[j]] / (4 * np.pi * d[index[j]] * d[index[j]] * 1e5))
				j = j + 1
				n = n + 1

				if(j == index.size):
					break

			S = np.append(S, temp1)
			temp1 = 0
			num = np.append(num, n)
			n = 0

		S = (S / (3.086e19 * 3.086e19)) * 1e26 # convert into Jansky
		delf_center = delf[:-1] + np.diff(delf)/2

		return delf_center, S, num

	# ------------------------------- #

	def find_angle(self, target):
		vec1 = target
		vec2 = np.array([self.x, self.y - 8.5, self.z])
		num = (vec1[0] * vec2[0]) + (vec1[1] * vec2[1]) + (vec1[2] * vec2[2]) 
		cos_angle = num / ( np.linalg.norm(vec1) * np.linalg.norm(vec2, axis = 0) )
		angle = np.arccos(cos_angle)
		return angle
	# ------------------------------- #


	def targetted_flux_density(self, doppf, angle):
		L = self.power_hooke()
		d = self.dist
		delf = np.arange(np.min(doppf), np.max(doppf), 1e-4)
		if(delf[-1] != np.max(doppf)):
			delf = np.append(delf, np.max(doppf))

		if(delf.size < 1):
			raise ValueError("First find doppler shifted frequency, then use this function.")


		index = self.referenced_sorting(angle, self.arrind)
		doppf_refs = self.referenced_sorting(angle, doppf)
		angle_sort = np.sort(angle)

		S = np.zeros(delf.size - 1)
		num = np.zeros(delf.size - 1, dtype = int)

		j, temp1, n  = 0, 0, 0
		m = np.min(doppf)

		sigma = self.beamFWHM / (np.sqrt(8 * np.log(2)))

		while(angle_sort[j] < self.beamFWHM/2):
			iii = int( (doppf_refs[j] - m) / (1e-4) )
			weight =  np.exp(- angle_sort[j] * angle_sort[j] / (2 * sigma * sigma) )
			flux =  (L[index[j]] / (4 * np.pi * d[index[j]] * d[index[j]] * 1e5)) * weight
			S[iii] = S[iii] + flux
			num[iii] = num[iii] + 1
			j = j+1


		S = (S / (3.086e19 * 3.086e19)) * 1e26 # convert into Jansky
		delf_center = delf[:-1] + np.diff(delf)/2

		return delf_center, S, num
