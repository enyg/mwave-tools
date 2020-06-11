#import sys
import yfactor as yf
import vna_import as vna
from yfactor import dBm_to_W
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
from matplotlib import pyplot as plt

class Component():
	def __init__(self, fspace, navg, name='', T_physical=290):
		self.fspace = fspace
		self.nf = None
		self.Teq = None
		self.gain = []
		self.oip3 = []
		self.iip3 = []
		self.navg = navg
		self.name = name
		self.Tp = T_physical
		# components default to resistive so lossy components will 
		# automatically have noise figure set appropriately based on gain
		self.resistive = True
	
	def setNF(self, fpoints, nfpoints):
		self.nf = np.interp(self.fspace, fpoints, nfpoints)
		self.Teq = (10**(self.nf/10)-1)*290
		self.resistive = False
		
	def setTeq(self, fpoints, Tpoints):
		self.Teq = np.interp(self.fspace, fpoints, Tpoints)
		self.nf = 10*np.log10(1 + self.Teq/290)
		self.resistive = False
		
	# calculate noise figure with simple y-factor method (no instrument calibration)
	# for LNAs the calibration might be important
	def calcNF(self, coldfile, hotfile, ENR_dB):
		# don't base noise figure on gain anymore after NF has been set this way
		self.resistive = False
		# import measurement data
		rbw, f_nf, p_hot = yf.loadcsv(hotfile)
		_, _, p_cold = yf.loadcsv(coldfile)
		# convert to Watts
		p_cold = dBm_to_W(p_cold)
		p_hot = dBm_to_W(p_hot)
		# smooth data
		if self.navg > 1:
			p_cold, trim = self.smooth(p_cold)
			p_hot, _ = self.smooth(p_hot)
			f_nf = f_nf[trim:-trim]
		# calculate equivalent noise temperature
		_, self.Teq = yf.yfactor_W(p_cold, p_hot, ENR_dB)
		# interpolate
		self.Teq = np.interp(self.fspace, f_nf, self.Teq)
		# convert to noise figure in dB
		F = self.Teq/290 + 1
		self.nf = 10*np.log10(F)
	
	# calculate noise figure and gain from y-factor method - overwrites existing gain
	def calcNFGain(self, coldfile_cal, hotfile_cal, coldfile, hotfile, ENR2dB, ENR12dB):
		# don't base noise figure on gain anymore after NF has been set this way
		self.resistive = False
		# import measurement data
		rbw, f_nf, p_hot = yf.loadcsv(hotfile)
		_, _, p_cold = yf.loadcsv(coldfile)
		_, fcal, cal_hot = yf.loadcsv(hotfile_cal)
		_, _, cal_cold = yf.loadcsv(coldfile_cal)
		# convert to Watts
		p_cold = dBm_to_W(p_cold)
		p_hot = dBm_to_W(p_hot)
		cal_cold = dBm_to_W(cal_cold)
		cal_hot = dBm_to_W(cal_hot)
		# smooth data
		if self.navg > 1:
			p_cold, trim = self.smooth(p_cold)
			p_hot, _ = self.smooth(p_hot)
			cal_cold, _ = self.smooth(cal_cold)
			cal_hot, _ = self.smooth(cal_hot)
			f_nf = f_nf[trim:-trim]
			fcal = fcal[trim:-trim]
		
		# make sure calibration frequency space is the same as DUT measurement
		if len(fcal) != len(f_nf):
			cal_cold = np.interp(f_nf, fcal, cal_cold)
			cal_hot = np.interp(f_nf, fcal, cal_hot)
		
		# calculate equivalent noise temperature
		nf, gain = yf.yfactor_cal_2ENR(cal_cold, cal_hot, p_cold, p_hot, ENR2dB, ENR12dB)
		self.Teq = (10**(nf/10)-1)*290
		# interpolate
		self.gain = np.interp(self.fspace, f_nf, gain)
		self.Teq = np.interp(self.fspace, f_nf, self.Teq)
		# convert to noise figure in dB
		F = self.Teq/290 + 1
		self.nf = 10*np.log10(F)
	
	# same as calcNFGain, but also corrects for input loss
	def calcNFGainIPLoss(self, coldfile_cal, hotfile_cal, coldfile, hotfile, ENR2dB, ENR12dB, iplossfile):
		

		# don't base noise figure on gain anymore after NF has been set this way
		self.resistive = False
		# import measurement data
		rbw, f_nf, p_hot = yf.loadcsv(hotfile)
		_, _, p_cold = yf.loadcsv(coldfile)
		_, _, cal_hot = yf.loadcsv(hotfile_cal)
		_, _, cal_cold = yf.loadcsv(coldfile_cal)
		# convert to Watts
		p_cold = dBm_to_W(p_cold)
		p_hot = dBm_to_W(p_hot)
		cal_cold = dBm_to_W(cal_cold)
		cal_hot = dBm_to_W(cal_hot)
		
		# input loss
		f2, sp = vna.loadcsv(iplossfile)
		iploss = np.abs(sp[:,2])
		iploss = np.interp(f_nf, f2, iploss)
		
		# smooth data
		if self.navg > 1:
			p_cold, trim = self.smooth(p_cold)
			p_hot, _ = self.smooth(p_hot)
			cal_cold, _ = self.smooth(cal_cold)
			cal_hot, _ = self.smooth(cal_hot)
			iploss, _ = self.smooth(iploss)
			f_nf = f_nf[trim:-trim]
		
		# convert input loss to db
		iploss = 20*np.log10(iploss)
		
		# calculate equivalent noise temperature
		nf, gain = yf.yfactor_cal_ipLoss(cal_cold, cal_hot, p_cold, p_hot, ENR2dB, ENR12dB, iploss)
		self.Teq = (10**(nf/10)-1)*290
		# interpolate
		self.gain = np.interp(self.fspace, f_nf, gain)
		self.Teq = np.interp(self.fspace, f_nf, self.Teq)
		# convert to noise figure in dB
		F = self.Teq/290 + 1
		self.nf = 10*np.log10(F)
	
	# frequency points for T_SA should match coldfile and hotfile
	def calcNFgainCal(self, coldfile, hotfile, ENR_dB, T_SA):
		# don't base noise figure on gain anymore after NF has been set this way
		self.resistive = False
		# import measurement data
		rbw, f_nf, p_hot = yf.loadcsv(hotfile)
		_, _, p_cold = yf.loadcsv(coldfile)
		# convert to Watts
		p_cold = dBm_to_W(p_cold)
		p_hot = dBm_to_W(p_hot)
		# smooth data
		if self.navg > 1:
			p_cold, trim = self.smooth(p_cold)
			p_hot, _ = self.smooth(p_hot)
			f_nf = f_nf[trim:-trim]
		# interpolate
		p_cold = np.interp(self.fspace, f_nf, p_cold)
		p_hot = np.interp(self.fspace, f_nf, p_hot)
		T_SA = np.interp(self.fspace, f_nf, T_SA)
		# calculate noise figure
		self.nf = yf.yfactor_cal_W2(p_cold, p_hot, T_SA, self.gain, ENR_dB)
		self.Teq = (10**(self.nf/10)-1)*290
	
	def setOIP3(self, fpoints, oip3points):
		self.oip3 = np.interp(self.fspace, fpoints, oip3points)
		if len(self.gain) != 0:
			self.iip3 = self.oip3 - self.gain
	
	def setGain(self, fpoints, gainpoints):
		# allow calling with just one gain value for the whole range
		if len(np.shape(gainpoints)) == 0:
			fpoints = [self.fspace[0]]
			gainpoints = [gainpoints]
			
		self.gain = np.interp(self.fspace, fpoints, gainpoints)
		
		# update NF
		if self.resistive == True:
			self.Teq = (10**(-self.gain/10)-1)*self.Tp
			self.nf = 10*np.log10(1 + self.Teq/290)
		
		# update IP3
		if len(self.oip3) != 0:
			self.iip3 = self.oip3 - self.gain
			
	def setGainFile(self, gainfile, column):
		# check if it's an s2p file or csv
		if gainfile[-4:] == '.s2p' or gainfile[-4:] == '.S2P':
			f1, SP = vna.loads2p(gainfile)
		else:
			f1, SP = vna.loadcsv(gainfile)
		
		gain = np.abs(SP[:,column])
		if self.navg > 1:
			gain, trim = self.smooth(gain)
			f1 = f1[trim:-trim]
		self.gain = 20*np.log10( np.interp(self.fspace, f1, gain) )
		
		# update NF
		if self.resistive == True:
			self.Teq = (10**(-self.gain/10)-1)*self.Tp
			self.nf = 10*np.log10(1 + self.Teq/290)
		
		# update IP3
		if len(self.oip3) != 0:
			self.iip3 = self.oip3 - self.gain
	
	def addGain(self, gainfile, column):
		f1, SP = vna.loadcsv(gainfile)
		gain = np.abs(SP[:,column])
		if self.navg > 1:
			gain, trim = self.smooth(gain)
			f1 = f1[trim:-trim]
		self.gain = self.gain + 20*np.log10( np.interp(self.fspace, f1, gain) )
		
		# update NF
		if self.resistive:
			self.Teq = (10**(-self.gain/10)-1)*self.Tp
			self.nf = 10*np.log10(1 + self.Teq/290)
		
		# update IP3
		if len(self.oip3) != 0:
			self.iip3 = self.oip3 - self.gain
		
	def gainRemoveAtten(self, attenfile, column):
		f1, SP = vna.loadcsv(attenfile)
		atten = np.abs(SP[:,column])
		if self.navg > 1:
			atten, trim = self.smooth(atten)
			f1 = f1[trim:-trim]
		self.gain = self.gain - 20*np.log10( np.interp(self.fspace, f1, atten))
		
		# update NF
		if self.resistive == True:
			self.Teq = (10**(-self.gain/10)-1)*self.Tp
			self.nf = 10*np.log10(1 + self.Teq/290)
		
		# update IP3
		if len(self.oip3) != 0:
			self.iip3 = self.oip3 - self.gain
		
	#def NFRemoveInputAdapter(self, adapterfile):
	
	def smooth(self, data):
		malen = self.navg
		# moving average filter
		avg = np.ones(malen)/malen
		dataf = np.convolve(data, avg, 'same')
		# get rid of the edges of the set since they get averaged with 0
		t = int((malen-1)/2)
		dataf = dataf[t:-t]
		return dataf, t

class Chain():
	def __init__(self, comp_list):
		self.components = comp_list
		self.Teq = 0
		self.G = 0
		self.NF = None
		self.IIP3 = []
		self.OIP3 = []
		self.name = ''
		self.calcCascade()
	
	def calcCascade(self):
		oip3_hi = 100	# number to use for linear components (dBm)
		iip3_w = 10**((oip3_hi-30)/10)
		
		self.Teq = 0
		self.G = 0
		
		for c in self.components:
			self.Teq = self.Teq + c.Teq/(10**(self.G/10))
			if len(c.oip3) == 0:
				c_oip3 = oip3_hi
			else:
				c_oip3 = c.oip3
			iip3_w = 1/(1/iip3_w + 10**((self.G-(c_oip3-30-c.gain))/10))
			self.G = self.G + c.gain
			#print(self.T_eq[0])
		
		# convert iip3 to dBm
		self.IIP3 = 10*np.log10(1000*iip3_w)
		self.OIP3 = self.IIP3 + self.G
		
		self.NF = 10*np.log10(self.Teq/290 + 1)
	
	# plot gains for all named components
	def plotGains(self):
		plt.figure()
		plt.rc('lines', linewidth=1)
		plt.title('Gain by component, \n' + self.name)
		plt.ylabel('Gain (dB)')
		plt.xlabel('Frequency (MHz)')
		for c in self.components:
			if c.name != '':
				plt.plot(c.fspace, c.gain, label=c.name)
		plt.plot(self.components[0].fspace, self.G, '--', label='Total gain')
		plt.legend()
		plt.grid()
	
	def plotTeqs(self):
		plt.figure()
		plt.rc('lines', linewidth=1)
		plt.title('$T_{eq}$ by component, \n' + self.name)
		plt.ylabel('Cascaded noise temp. (K)')
		plt.xlabel('Frequency (MHz)')
		num = 0
		for c in self.components:
			num = num + 1
			if c.name != '':
				partial = Chain(self.components[:num])
				plt.plot(c.fspace, partial.Teq, label='after '+c.name)
		plt.plot(self.components[0].fspace, self.Teq, '--', label='Overall $T_{eq}$')
		plt.legend()
		plt.grid()
	
	# plot OIP3 for all named components
	def plotOIP3s(self):
		plt.figure()
		plt.rc('lines', linewidth=1)
		plt.title('$OIP_3$ by component, \n' + self.name)
		plt.ylabel('OIP3 (dBm)')
		plt.xlabel('Frequency (MHz)')
		for c in self.components:
			if (c.name != '') and (len(c.oip3) != 0):
				plt.plot(c.fspace, c.oip3, label=c.name)
		plt.plot(self.components[0].fspace, self.OIP3, '--', label='Cascaded OIP3')
		plt.legend()
		plt.grid()
	
	# plot IIP3 for all named components
	def plotIIP3s(self):
		plt.figure()
		plt.rc('lines', linewidth=1)
		plt.title('$IIP_3$ by component, \n' + self.name)
		plt.ylabel('IIP3 (dBm)')
		plt.xlabel('Frequency (MHz)')
		for c in self.components:
			if (c.name != '') and (len(c.oip3) != 0):
				plt.plot(c.fspace, c.iip3, label=c.name)
		plt.plot(self.components[0].fspace, self.IIP3, '--', label='Cascaded IIP3')
		plt.legend()
		plt.grid()
		
	# print the noise budget at a single frequency
	def printBudget(self, freq):
		print("Noise Budget at " + str(freq) + " MHz:")
		print(" " + self.name)
		
		num = 0
		for c in self.components:
			num = num + 1
			partial = Chain(self.components[:num])
			
			gain = str(round(np.interp(freq, c.fspace, c.gain),2))
			nf = str(round(np.interp(freq, c.fspace, c.nf),2))
			Teq = str(round(np.interp(freq, c.fspace, c.Teq),2))
			Tsys = str(round(np.interp(freq, c.fspace, partial.Teq),2))
			print(" " + c.name + " "*(20-len(c.name)) + gain + " dB      "
				+ nf + " dB / " + Teq + " K       " 
				+ str(c.Tp) + " K phys     " 
				+ Tsys + " K total")
	
	# use formulas to estimate rfi intermods (faster than time domain method for large cascades)
	def intermodEstimate(self, rfi_freq, rfi_bw, rfi_pow, t):
		
		#set up frequency space
		dt = min(np.diff(t))
		Fs = 1/dt
		fmin = -1/(2*dt)
		fmax = 1/(2*dt)
		fnl = np.linspace(0, fmax, int(len(t)/2))
		df = Fs/len(t)
		
		rfi_freq = np.concatenate((-np.array(rfi_freq), np.array(rfi_freq)))
		rfi_bw = np.concatenate((-np.array(rfi_bw), np.array(rfi_bw)))
		rfi_pow = dBm_to_W(np.array(rfi_pow))
		rfi_pow = np.concatenate((rfi_pow, rfi_pow))
		n = len(rfi_freq)
		
		# initialize output spectrum (input state)	(1-sided spectrum)
		VSDout = np.zeros(int(len(t)/2))
		#S_out_est = np.ones(int(len(t)/2)) * dBm_to_W(-200)
		# rfi source input power
		for ix in range(len(rfi_freq)):
			if rfi_freq[ix] > 0:
				rng_fund = (np.where((fnl >= rfi_freq[ix]-rfi_bw[ix]/2) & (fnl <= rfi_freq[ix]+rfi_bw[ix]/2)))[0]
				VSDout[rng_fund] = np.sqrt(2*50*rfi_pow[ix]/np.abs(rfi_bw[ix]))
				#S_out_est[rng_fund] = rfi_pow[ix]/np.abs(rfi_bw[ix])
		
		plot=False
		if plot:
			plt.figure()
			plt.title('component contributions, estimate')
			
		for c in self.components:
			# calculate power series gain coefficients from this component's gain/OIP3
			a1f = 10**(c.gain/20)
			
			# apply gain to whole spectrum in this case
			# it's ok that this is also affecting the fundamental before that's applied to IM power calculation,
			# because that calculation is using the array of RFI powers anyway (not the voltage spectrum)
			a1 = np.interp(fnl, c.fspace*1e6, a1f)
			VSDout = VSDout*a1
			
			if len(c.oip3) == 0:
				#c_oip3 = oip3_hi
				a3_est = []	# just set a3 to zero since we're modeling these devices as linear
				
				#S_out_est = S_out_est*a1**2
			else:
				#c_oip3 = c.oip3
				a3_est = 2*10**(1.5*c.gain/10)/(3*dBm_to_W(c.oip3)*50)

				a3f = a3_est

				# for every combination of 3 rfi sources
				for i1 in range(n):
					for i2 in range(n):
						for i3 in range(n):
							# sum frequency
							fsum = rfi_freq[i1] + rfi_freq[i2] + rfi_freq[i3]
							# calculate power for positive frequencies
							if fsum > 0:
								a3 = np.interp(fsum, c.fspace*1e6, a3f)
								P = (50/2*a3)**2 * rfi_pow[i1] * rfi_pow[i2] * rfi_pow[i3] 
								# calculate bandwidth
								BW = np.abs(rfi_bw[i1] + rfi_bw[i2] + rfi_bw[i3])
								
								# modify output spectrum							
								rng_im = (np.where((fnl >= fsum-BW/2) & (fnl <= fsum+BW/2)))[0]
								VSDout[rng_im] = VSDout[rng_im] + np.sqrt(2*50*P/BW)
								#S_out_est[rng_im] = S_out_est[rng_im] + P/np.abs(BW)
				
				if plot and (c.name != '') and (len(c.oip3) != 0):
					Sp = VSDout**2/(2*50) + dBm_to_W(-300)
					plt.plot(fnl*1e-6, 30+10*np.log10(Sp), '--', label='output after '+c.name)
					
				
				# calculate fundamental gain from this component and update plot
				# ~ for ix in range(len(rfi_freq)):
					# ~ if rfi_freq[ix] > 0:
						# ~ a1 = np.interp(rfi_freq[ix], c.fspace*1e6, a1f)
						# ~ rng_fund = (np.where((fnl >= rfi_freq[ix]-rfi_bw[ix]/2) & (fnl <= rfi_freq[ix]+rfi_bw[ix]/2)))[0]
						# ~ VSDout[rng_fund] = VSDout[rng_fund]*a1
						#S_out_est[rng_fund] = S_out_est[rng_fund]*a1**2
			
			# update rfi source power with component gain
			for ix in range(len(rfi_freq)):	
				a1 = np.interp(rfi_freq[ix], c.fspace*1e6, a1f)				
				rfi_pow[ix] = rfi_pow[ix]*a1**2
			
		# find peak regions for plotting (max intermod?)
		
		if plot:
			plt.legend(loc='upper right')
		
		S_out_est = VSDout**2/(2*50) + dBm_to_W(-300)
		
		return fnl, S_out_est
	
	# wrapper if you don't need the voltage output
	def timeDomainStepByStep(self, t, vin):
		f, S_out, S3, S_in, _ = self.timeDomainStepByStepV(t, vin)
		return f, S_out, S3, S_in
	
	# calculate response of each component separately
	def timeDomainStepByStepV(self, t, vin):
		fspace = self.components[0].fspace
		
		plot = False
		if plot:
			plt.figure()
			plt.title('component nonlinear contributions')
		
		#set up frequency space
		dt = min(np.diff(t))
		Fs = 1/dt
		NFFT = len(vin)		
		fmin = -1/(2*dt)
		fmax = 1/(2*dt)
		f = np.linspace(0, fmax, int(NFFT/2))
		df = Fs/NFFT
		
		S_in = psd_1sided(vin, Fs, NFFT)
		S_in = S_in[int(len(vin)/2):]
		
		Vout = vin
		#v1sum = np.zeros(len(vin))
		v3sum = np.zeros(len(vin))
		vout3 = np.zeros(len(vin))
		
		oip3_hi = 100	# number to use for linear components (dBm)
		iip3_w = 10**((oip3_hi-30)/10)
		
		for c in self.components:
			# calculate power series gain coefficients from this component's gain/OIP3
			a1f = 10**(c.gain/20)
			
			if len(c.oip3) == 0:
				#c_oip3 = oip3_hi
				a3_est = []	# just set a3 to zero since we're modeling these devices as linear
			else:
				#c_oip3 = c.oip3
				a3_est = 2*10**(1.5*c.gain/10)/(3*dBm_to_W(c.oip3)*50)
			
			a3f = a3_est
			
			# Vout is fed back in to the next component
			vout1, vout3 = timeDomainVoltage(Vout, a1f, a3f, Fs, fspace)
			
			Vout = vout1 + vout3
			#v1sum = v1sum + vout1
			v3sum = v3sum + vout3
			
			S3 = psd_1sided(vout3, Fs, NFFT)
			S3 = S3[int(len(vin)/2):]
			
			S1 = psd_1sided(vout1, Fs, NFFT)
			S1 = S1[int(len(vin)/2):]
			
			if plot and (c.name != '') and (len(c.oip3) != 0):
				plt.plot(f*1e-6, 30+10*np.log10(S3), ':', label='3rd order (after '+c.name+')')
				plt.plot(f*1e-6, 30+10*np.log10(S1), label='linear (after '+c.name+')')
		
		if plot:
			plt.plot(f*1e-6, 30+10*np.log10(S_in), label='input')
			plt.legend(loc="upper right")
			plt.xlabel("frequency (MHz)")
			plt.ylabel("dBm/Hz")
		
		S_out = psd_1sided(Vout, Fs, NFFT)
		S_out = S_out[int(len(vin)/2):]
		
		S3 = psd_1sided(v3sum, Fs, NFFT)
		S3 = S3[int(len(vin)/2):]
		
		return f, S_out, S3, S_in, Vout
			
	# frequency dependent gain and a3
	def timeDomainResponse(self, t, vin):
		if len(self.OIP3) == 0:
			self.calcCascade()
		
		fspace = self.components[0].fspace
		
		# calculate power series gain coefficients from gain / OIP3
		
		a1f = 10**(self.G/20)
		# TODO: use direct a3 measurements if available
		a3_est = 2*10**(1.5*self.G/10)/(3*dBm_to_W(self.OIP3)*50)
		a3f = a3_est
		
		# calculate nonlinear response
		dt = min(np.diff(t))
		Fs = 1/dt
		NFFT = len(vin)		
		fmin = -1/(2*dt)
		fmax = 1/(2*dt)
		f = np.linspace(0, fmax, int(NFFT/2))
		df = Fs/NFFT

		S_in = psd_1sided(vin, Fs, NFFT)
		S_in = S_in[int(len(vin)/2):]
		
		p_tot = sum(S_in) * df
		p_tot_dbm = 10*np.log10(1000*p_tot)
		print("total input power (dBm): ", p_tot_dbm)
		
		vout1, vout3 = timeDomainVoltage(vin, a1f, a3f, Fs, fspace)
		
		## debug:
		# ~ plt.figure()
		# ~ plt.title('a3 voltage gain from component')
		# ~ plt.plot(fspace, a3f)
		
		# ~ plt.figure()
		# ~ plt.plot(ff, vin_f, label='input fft')
		# ~ plt.plot(ff, vin_f*A1f, label='output linear')
		# ~ plt.plot(ff, fft(vin**3)*A3f, label='output 3rd order') 
		# ~ plt.plot(ff, A3f, label='a3 gain')
		# ~ plt.xlabel("frequency (Hz)")
		# ~ plt.ylabel("V/sqrt(Hz)")
		# ~ plt.legend()
		
		# ~ S_out_lin = psd_1sided(vout1, Fs, NFFT)
		# ~ S_out_lin = S_out_lin[int(len(vin)/2):]
		
		S3 = psd_1sided(vout3, Fs, NFFT)
		S3 = S3[int(len(vin)/2):]
		
		# ~ plt.figure()
		# ~ plt.plot(f, 30+10*np.log10(S_in), label='input psd')
		# ~ plt.plot(f, 30+10*np.log10(S_out_lin), label='linear output psd')
		# ~ plt.plot(f, 30+10*np.log10(S3), label='3rd order output psd')
		# ~ plt.legend()
		
		#plt.show()
		
		# sum 1st and 3rd order outputs
		vout = vout1 + vout3
		
		S_out = psd_1sided(vout, Fs, NFFT)
		S_out = S_out[int(len(vin)/2):]
		
		peak_psd = 10*np.log10(1000*max(S_out))
		p_tot = sum(S_out) * df
		p_tot_dbm = 10*np.log10(1000*p_tot)
		print("total output power (dBm): ", p_tot_dbm)
		
		return f, S_out, S3, S_in
		
	def timeDomainResponseWC(self, t, vin):
		if len(self.OIP3) == 0:
			self.calcCascade()
				
		# calculate power series gain coefficients from gain / OIP3
		
		# for now, use the worst case OIP3
		# TODO: take into account the OIP3 varying over frequency
		roi = (np.where((self.components[0].fspace >= 1300) & (self.components[0].fspace <= 1720)))[0]
		oip3 = min(self.OIP3[roi])
		gain = min(self.G[roi])
		
		print('worst case OIP3: ', oip3)
		print('worst case gain: ', gain)
		
		Z0 = 50
		a1 = 10**(gain/20)
		a3 = -2*a1**3/(3*Z0*yf.dBm_to_W(oip3))
		
		# calculate nonlinear response
		dt = min(np.diff(t))
		Fs = 1/dt
		NFFT = len(vin)		
		fmin = -1/(2*dt)
		fmax = 1/(2*dt)
		f = np.linspace(0, fmax, int(NFFT/2))
		df = Fs/NFFT

		S_in = psd_1sided(vin, Fs, NFFT)
		S_in = S_in[int(len(vin)/2):]
		
		p_tot = sum(S_in) * df
		p_tot_dbm = 10*np.log10(1000*p_tot)
		print("total input power (dBm): ", p_tot_dbm)
		
		vout3 = a3*vin**3
		
		vout = a1*vin + vout3
		S_out = psd_1sided(vout, Fs, NFFT)
		S_out = S_out[int(len(vin)/2):]
		
		S3 = psd_1sided(vout3, Fs, NFFT)
		S3 = S3[int(len(vin)/2):]
		
		peak_psd = 10*np.log10(1000*max(S_out))
		p_tot = sum(S_out) * df
		p_tot_dbm = 10*np.log10(1000*p_tot)
		print("total output power (dBm): ", p_tot_dbm)
		
		return f, S_out, S3, S_in

def timeDomainVoltage(vin, a1f, a3f, Fs, fspace):
	# linear output term
	vin_f = fft(vin)
	ff = fftfreq(len(vin)) # in cycles/sample?
	ff = Fs*ff	# convert to cycles/sec
	fpos = np.arange(0,int(len(vin)/2)+1)
	fneg = np.arange(int(len(vin)/2)+1,len(vin))	# not sure about these
	# interpolate a1(f) to positive frequency space of fft
	A1fp = np.interp(ff[fpos], fspace*1e6, a1f)
	A1fn = np.interp(ff[fneg], -np.flip(fspace)*1e6, np.flip(a1f))
	A1f = np.concatenate((A1fp, A1fn))
	vout1 = ifft(vin_f*A1f)
	
	# save time if this is a linear computation
	if len(a3f)==0:
		vout3 = np.zeros(len(vin))
	else:
		# compute ifft of a3(f) to convolve with vin^3
		A3fp = np.interp(ff[fpos], fspace*1e6, a3f)
		A3fn = np.interp(ff[fneg], -np.flip(fspace)*1e6, np.flip(a3f))
		A3f = np.concatenate((A3fp, A3fn))
		a3t = ifft(A3f)
		# this is inefficient because it's doing the fft(ifft(b)), so I could do real(ifft(fft(vin**3)*A3f)) instead
		# circular convolution because this is discrete (I think that's the reason) and we need valid values over the whole spectrum
		vout3 = np.real(ifft(fft(vin**3)*A3f))
		#vout3 = conv_circ(a3t, vin**3)
	
	return vout1, vout3
		
def psd_1sided(vt, Fs, NFFT):
	# window
	win = np.ones(len(vt))
	S = sum(win**2)	# scaling factor to account for window
	dft = fftshift(fft(vt*win, NFFT))
	p = 2/(50*S*Fs)*np.abs(dft)**2 # factor of 2 is to include power at negative frequencies
	
	return p
	
def conv_circ(a, b):
	# a and b need to have the same shape
	return np.real(ifft(fft(a)*fft(b)))
