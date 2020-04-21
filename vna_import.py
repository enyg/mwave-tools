# load s-parameter files from VNA in csv format

import numpy as np
import csv

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
