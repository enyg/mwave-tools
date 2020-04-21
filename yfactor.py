## Calculate noise figure from Agilent N9320B SA exported data
## Erich Nygaard, BYU Radio Astronomy
## Parts based on Dave Buck's MATLAB yfactor script
## 3/20/2020

import numpy as np
import csv
import re

def loadcsv(f):
	# compile regex to extract numbers
	p = re.compile("[^a-df-zA-Z]*")
	
	rbw = 0
	
	with open(f, 'r', encoding='utf-8', errors='replace') as csvfile:
		nfreader = csv.reader(csvfile, delimiter=',')
		for row in nfreader:
			if len(row) > 0:
				if row[0] == "Resolution Bandwidth:":
					rbw = float(row[1])
					if row[2] == 'kHz':
						rbw = rbw*1000
					elif row[2] == 'MHz':
						rbw = rbw*1000000
					#print('RBW', rbw, 'Hz')
			if len(row) > 0 and row[0] == 'No.':
				break
		
		freq = []
		power = []
		for row in nfreader:
			if len(row) == 3:
				freq.append(float(p.match(row[1]).group()))
				power.append(float(p.match(row[2]).group()))
		freq = np.array(freq)/1e6
		power = np.array(power)
		return rbw, freq, power
				
## convenience functions
T0 = 290

def dBm_to_W(PdBm):
	return .001*10**(PdBm/10.0)

def dB_to_real(dB):
	return 10**(dB/10.)

def NF_to_T(NFdB):
	return T0*(10**(NFdB/10.) - 1)

## Calculation methods

# simple method
def yfactor(pcold, phot, ENR_dB):
	ENR = dB_to_real(ENR_dB)
	Y = dBm_to_W(phot) / dBm_to_W(pcold)
	F = ENR / (Y - 1)
	Tsys = (F - 1)*T0
	#print('Y',Y,'\nF',F)
	NF = 10*np.log10(F)
	return NF
	
# simple method, Watt input instead of dBm
def yfactor_W(pcold, phot, ENR_dB):
	ENR = dB_to_real(ENR_dB)
	Y = phot / pcold
	F = ENR / (Y - 1)
	Tsys = (F - 1)*T0
	#print('Y',Y,'\nF',F)
	NF = 10*np.log10(F)
	return NF, Tsys

# full cal yfactor method
def yfactor_cal_W(N2_off, N2_on, N12_off, N12_on, ENR_dB):
	ENR = dB_to_real(ENR_dB)
	
	# SA noise temperature(T2)
	
	Ts_off_F = 70 # Physical temp of noise source at time of measurement
	Ts_off = (Ts_off_F-32)/1.8 + 273.15 # Kelvin
	
	Ts_on = ENR*T0+Ts_off
	
	Y2 = N2_on/N2_off
	T2 = (Ts_on + Y2*Ts_off)/(Y2-1)
	
	# Combined DUT + SA noise temperature (T12)
	Y12 = N12_on/N12_off
	T12 = (Ts_on - Y12*Ts_off)/(Y12-1)
	
	# DUT Gain
	G1 = (N12_on - N12_off)/(N2_on - N2_off)
	
	# DUT noise temperature (T1)
	T1 = T12 - (T2/G1)
	F1 = 1 + T1/T0
	NF1 = 10*np.log10(F1)
	
	return NF1, 10*np.log10(G1)

# full cal yfactor method - Temp of SA and gain already known
def yfactor_cal_W2(N12_off, N12_on, T2, G1dB, ENR_dB):
	ENR = dB_to_real(ENR_dB)
	#ENR_SA = dB_to_real(ENR_SA_dB)
	
	# SA noise temperature(T2)
	
	Ts_off_F = 70 # Physical temp of noise source at time of measurement
	Ts_off = (Ts_off_F-32)/1.8 + 273.15 # Kelvin
	
	Ts_on = ENR*T0+Ts_off
	#Y2 = N2_on/N2_off
	#T2 = (Ts_on + Y2*Ts_off)/(Y2-1)
	
	# Combined DUT + SA noise temperature (T12)
	Y12 = N12_on/N12_off
	T12 = (Ts_on - Y12*Ts_off)/(Y12-1)
	
	# DUT Gain
	#G1 = (N12_on - N12_off)/(N2_on - N2_off)
	G1 = 10**(G1dB/10)
	
	# DUT noise temperature (T1)
	T1 = T12 - (T2/G1)
	F1 = 1 + T1/T0
	NF1 = 10*np.log10(F1)
	
	return NF1

# full cal y factor method, but calibration was done with a different ENR source	
def yfactor_cal_2ENR(N2_off, N2_on, N12_off, N12_on, ENR2dB, ENR12dB):
	ENR2 = dB_to_real(ENR2dB)
	ENR12 = dB_to_real(ENR12dB)
	# SA noise temperature(T2)
	
	Ts_off_F = 70 # Physical temp of noise source at time of measurement
	Ts_off = (Ts_off_F-32)/1.8 + 273.15 # Kelvin
	
	Ts_on = ENR2*T0+Ts_off
	Y2 = N2_on/N2_off
	T2 = (Ts_on + Y2*Ts_off)/(Y2-1)
	
	# Combined DUT + SA noise temperature (T12)
	Ts_on = ENR12*T0 + Ts_off
	Y12 = N12_on/N12_off
	T12 = (Ts_on - Y12*Ts_off)/(Y12-1)
	
	# DUT Gain
	G1 = (N12_on - N12_off)/(N2_on - N2_off)*(ENR2/ENR12)
	
	# DUT noise temperature (T1)
	T1 = T12 - (T2/G1)
	F1 = 1 + T1/T0
	NF1 = 10*np.log10(F1)
	
	return NF1, 10*np.log10(G1)

# for non-reflective losses (attenuators etc)
def yfactor_cal_ipLoss(N2_off, N2_on, N12_off, N12_on, ENR2dB, ENR12dB, ipLossdB):
	ENR2 = dB_to_real(ENR2dB)
	ENR12 = dB_to_real(ENR12dB)
	# SA noise temperature(T2)
	
	Ts_off_F = 70 # Physical temp of noise source at time of measurement
	Ts_off = (Ts_off_F-32)/1.8 + 273.15 # Kelvin
	
	Ts_on = ENR2*T0+Ts_off
	Y2 = N2_on/N2_off
	T2 = (Ts_on + Y2*Ts_off)/(Y2-1)
	
	# Combined DUT + SA noise temperature (T12)
	Ts_on = ENR12*T0 + Ts_off
	Y12 = N12_on/N12_off
	T12 = (Ts_on - Y12*Ts_off)/(Y12-1)
	
	# input loss correction
	Lin = 10**(-ipLossdB/10) # ratio greater than 1
	T12in = T12/Lin - (Lin-1)*Ts_off/Lin
	
	# DUT Gain
	G1 = (N12_on - N12_off)/(N2_on - N2_off)*(ENR2/ENR12)
	# correct for input loss
	G1in = G1*Lin
	
	# DUT noise temperature (T1)
	T1 = T12in - (T2/G1in)
	F1 = 1 + T1/T0
	NF1 = 10*np.log10(F1)
	
	return NF1, 10*np.log10(G1in)

def yfactor_new(N2_off, N2_on, N12_off, N12_on, ENR_dB):
	ENR = dB_to_real(ENR_dB)
	
	# SA noise temperature(T2)
	
	Ts_off_F = 70 # Physical temp of noise source at time of measurement
	Ts_off = (Ts_off_F-32)/1.8 + 273.15 # Kelvin
	
	Ts_on = ENR*T0+Ts_off
	
	Y2 = N2_on/N2_off
	T2 = (Ts_on + Y2*Ts_off)/(Y2-1)
	
	# Combined DUT + SA noise temperature (T12)
	Y12 = N12_on/N12_off
	T12 = (Ts_on - Y12*Ts_off)/(Y12-1)
	
	# DUT Gain
	GN = (N12_on - N12_off)/(N2_on - N2_off)
	
	# DUT noise temperature (T1)
	T1 = (T12 - (T2/GN))/(1+T2/GN/T0)
	F1 = 1 + T1/T0
	NF1 = 10*np.log10(F1)
	
	G1 = GN/F1
	
	return NF1, 10*np.log10(G1)

# spectrum analyzer noise figure
def NF_SA(N2_off, N2_on, ENR_dB):
	ENR = dB_to_real(ENR_dB)
	
	# SA noise temperature(T2)
	Ts_off_F = 70 # Physical temp of noise source at time of measurement
	Ts_off = (Ts_off_F-32)/1.8 + 273.15 # Kelvin
	
	Ts_on = ENR*T0+Ts_off
	
	Y2 = N2_on/N2_off
	T2 = (Ts_on + Y2*Ts_off)/(Y2-1)
	F2 = 1 + T2/T0
	NF2 = 10*np.log10(F2)
	return NF2, T2

# smooth the input data first
def smooth(data, malen):
	# moving average filter
	avg = np.ones(malen)/malen
	dataf = np.convolve(data, avg, 'same')
	# get rid of the edges of the set since they get averaged with 0
	t = int((malen-1)/2)
	dataf = dataf[t:-t]
	return dataf
