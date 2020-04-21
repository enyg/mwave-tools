#import sys
#sys.path.insert(1, '/home/erich/data/research/RAS/py_modules')
import yfactor as yf
import vna_import as vna
from yfactor import dBm_to_W
import numpy as np

class Component():
	def __init__(self, fspace, navg):
		self.fspace = fspace
		self.nf = None
		self.Teq = None
		self.gain = []
		self.oip3 = []
		self.iip3 = []
		self.navg = navg
		# components default to resistive so lossy components will 
		# automatically have noise figure set appropriately based on gain
		self.resistive = True
	
	def setNF(self, fpoints, nfpoints):
		self.nf = np.interp(self.fspace, fpoints, nfpoints)
		self.Teq = (10**(self.nf/10)-1)*290
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
		self.gain = np.interp(self.fspace, fpoints, gainpoints)
		
		# update NF
		if self.resistive == True:
			self.nf = -self.gain
			self.Teq = (10**(self.nf/10)-1)*290
		
		# update IP3
		if len(self.oip3) != 0:
			self.iip3 = self.oip3 - self.gain
			
	def setGainFile(self, gainfile, column):
		f1, SP = vna.loadcsv(gainfile)
		gain = np.abs(SP[:,column])
		if self.navg > 1:
			gain, trim = self.smooth(gain)
			f1 = f1[trim:-trim]
		self.gain = 20*np.log10( np.interp(self.fspace, f1, gain) )
		
		# update NF
		if self.resistive == True:
			self.nf = -self.gain
			self.Teq = (10**(self.nf/10)-1)*290
		
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
			self.nf = -self.gain
			self.Teq = (10**(self.nf/10)-1)*290
		
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
			self.nf = -self.gain
			self.Teq = (10**(self.nf/10)-1)*290
		
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
