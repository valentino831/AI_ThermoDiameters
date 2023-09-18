# -*- coding: utf-8 -*-

# modules for math and plotting
import numpy as np        
from matplotlib import pyplot as plt  
from scipy import ndimage


# import the module
import fnv
import fnv.reduce
import fnv.file   

from datetime import datetime

import glob
import os
import csv

class FlirVideo:
    def __init__(self, fileName):
        self.im = fnv.file.ImagerFile(fileName)
        self.Temp = np.empty([self.im.height, self.im.width, self.im.num_frames])
        self.time = np.empty(self.im.num_frames)

        self.im.get_frame(0)
        
        data_0 = self.im.frame_info.time

        if self.im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
            # set units to temperature, if available
            self.im.unit = fnv.Unit.TEMPERATURE_FACTORY
            self.im.temp_type = fnv.TempType.CELSIUS         # set temperature unit
        else:
            # if file has no temperature calibration, use counts instead
            self.im.unit = fnv.Unit.COUNTS


        for i in range(self.im.num_frames):
            self.im.get_frame(i)                         # get the current frame

            self.time[i] = (self.im.frame_info.time-data_0).total_seconds()

            # convert image to np array
            # this makes it easy to find min/max
            self.Temp[...,i] = np.array(self.im.final, copy=False).reshape(
                (self.im.height, self.im.width))

    def findExcitmentPeriod(self, nFrame):

        idx = np.where(self.Temp==self.Temp.max())
        # plt.plot(fv.time, fv.Temp[idx[0][0], idx[1][0],:])
        t = fv.Temp[idx[0][0], idx[1][0],:].reshape(self.time.shape)

        idxIni = np.where(t>=3.5*np.sqrt(np.var(t[:nFrame]))+np.mean(t[:nFrame]))
        
        ii = idxIni[0][0]
        jj = ii
        fine = False
        while not(fine):
            zz = ii+nFrame
            if zz>t.size:
                zz = t.size
            
            t1 = t[ii:zz]
            idxFine = np.where(t1==t1.max())
            idxFine = idxFine[0][0]+ii-1

            if t[idxFine] >= t[jj]:
                jj = idxFine
            else:
                fine = True

            ii = zz
    
        return [idxIni[0][0], jj]

    def videoCut(self, idxs):
        self.Temp = self.Temp[:,:,idxs[0]:idxs[1]]
        self.time = self.time[idxs[0]:idxs[1]]
        self.time = self.time-self.time[0]

    def generateRandomData(self, fileName):
        angle = 360*np.random.random()
        Temp = self.Temp
        Temp = ndimage.rotate(Temp, angle=angle, reshape=False, mode='nearest')
        if np.random.random()>=0.5:
            Temp = np.fliplr(Temp)
        if np.random.random()>=0.5:
            Temp = np.flipud(Temp)
        Temp = Temp+0.5*(np.random.random(Temp.shape)-0.5)

        np.savez_compressed(fileName, Temp=Temp, time=self.time)

    def saveTemp(self, fileName):
        Temp = self.Temp

        np.savez_compressed(fileName, Temp=Temp, time=self.time)


path = '/media/valentino/OS/Users/d016781/Dropbox (Politecnico Di Torino Studenti)/Termografia_Santoro/Razza_0210/'

file_list = glob.glob(path + '*.ats')

with open('test.csv', 'w') as f:
    csvWriter = csv.writer(f)
    for file in file_list:
        print(os.path.splitext(os.path.basename(file))[0])
        fv = FlirVideo(file)

        print(fv.findExcitmentPeriod(100))

        fv.videoCut(fv.findExcitmentPeriod(100))

        for i in range(20):
            fv.generateRandomData('train/' + os.path.splitext(os.path.basename(file))[0] + '_' + str(i) + '.npz')
            # csvWriter.writerow('train/' + os.path.splitext(os.path.basename(file))[0] + '_' + str(i) + '.npz')
        
        # fv.saveTemp('test/' + os.path.splitext(os.path.basename(file))[0] + '.npz')

# loaded = np.load('file.npz')
# Temp = loaded['Temp']
# time = loaded['time']

