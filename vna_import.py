# load s-parameter files from VNA in csv format

import numpy as np
import csv, re

def loadcsv(f):
	with open(f, 'r', encoding='utf-8') as csvfile:
		nfreader = csv.reader(csvfile, delimiter=',')
		for row in nfreader:
			if len(row) > 0:
				if row[0] == 'Freq(Hz)':
					#ports = int(np.sqrt((len(row)-1)/2))
					cols = len(row)
					break
		freq = []
		nports = int((cols-1)/2)
		SP = np.zeros([0,nports])
		for row in nfreader:
			if len(row) == cols:
			#if len(row) == ports**2*2+1:
				freq.append(float(row[0]))
				sp = np.zeros([1,nports], dtype=complex) # ports**2
				for c in range(nports):	# ports**2
					sp[0,c] = float(row[1+c*2]) + 1j*float(row[2+c*2])
				SP = np.concatenate((SP, sp))
						
		freq = np.array(freq)/1e6
		return freq, SP

# reads s2p files that are in real/imag format (or dB / angle - but ignore angle)
def loads2p(f):
	freq_prefix = 1
	
	fmt = ''
	
	with open(f, 'r', encoding='utf-8') as s2pfile:
		s2pReader = csv.reader(s2pfile, delimiter=' ', skipinitialspace=True)
		for row in s2pReader:
			if len(row) > 0:
				if row[0] == '#':
					# header row
					if row[1] == 'GHz':
						freq_prefix = 1e9
					if row[1] == 'MHz':
						freq_prefix = 1e6
					if row[1] == 'kHz' or row[1] == 'KHz':
						freq_prefix = 1e3
					if row[3] == 'RI':
						fmt = 'RI'
					elif row[3] == 'dB':
						fmt = 'dB'
					assert(row[3] == 'RI' or row[3] == 'dB')
					break
		
		# determine number of ports (columns)
		for row in s2pReader:
			if row[1] == 'Freq':
				cols = len(row)-1
				break
		nports = int((cols-1)/2)
		
		freq = []
		SP = np.zeros([0,nports])
		
		for row in s2pReader:
			if len(row) == 1:
				row = re.split('\s', row[0])	# for weird files that don't use spaces
			if len(row) == cols:
				freq.append(float(row[0]))
				sp = np.zeros([1,nports], dtype=complex)
				for c in range(nports):
					if fmt == 'RI':
						sp[0,c] = float(row[1+c*2]) + 1j*float(row[2+c*2])
					elif fmt == 'dB':
						sp[0,c] = 10**(float(row[1+c*2])/20)	# just get the real part for now
				SP = np.concatenate((SP, sp))
			#else:
			#	print(len(row), '!=', cols)
		
		freq = np.array(freq)*freq_prefix/1e6
		return freq, SP
