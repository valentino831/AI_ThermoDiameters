import numpy as np        
from scipy import ndimage

# import the module
import fnv
import fnv.reduce
import fnv.file   


class FlirVideo:
    def __init__(self, fileName):
        # print(fileName)
        self.im = fnv.file.ImagerFile(fileName)
        self.Temp = np.empty([self.im.height, self.im.width, self.im.num_frames])
        self.time = np.empty(self.im.num_frames)
        self.Tempbatshe = 0
        
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
        # plt.plot(self.time, self.Temp[idx[0][0], idx[1][0],:])
        t = self.Temp[idx[0][0], idx[1][0],:].reshape(self.time.shape)

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
        
    def videoCutbatch(self, SEQ_LENGTH):
        n = self.im.num_frames % SEQ_LENGTH
        if n/SEQ_LENGTH > 0.8:
            arr2 = np.zeros((self.im.height, self.im.width, int((SEQ_LENGTH -n))))
            self.Temp = np.concatenate((self.Temp, arr2), axis=2)
        else :       
            self.Temp = self.Temp[:,:,:-n]
        s  = np.shape(self.Temp)[-1] / SEQ_LENGTH
        self.Tempbatshe = np.split(self.Temp,s,axis=2)
            
        
       
        #self.Temp-batshe = np.empty([self.im.height, self.im.width, self.im.num_frames,SEQ_LENGTH])
        #self.time-batche = np.empty(self.im.num_frames)
        #self.Temp = self.Temp[:,:,idxs[0]:idxs[1]]
        #self.time = self.time[idxs[0]:idxs[1]]
        #self.time = self.time-self.time[0]
    def load_video(self, max_frames=0, resize=(136,144)):
        Temp = self.Temp
        min = Temp.min()
        max = Temp.max()

        Temp = 255*(Temp-min)/(max-min)

        frames = []
        len = Temp.shape[2]
        
        for i in range(len):
            size = Temp.shape

            if size[0] < resize[0]:
                d1 = np.int16(np.floor((resize[0]-size[0])/2))
                d2 = np.int16(resize[0]-size[0]-d1)

                T = np.append(np.zeros((d1, size[1])), np.append(Temp[:,:, i], np.zeros((d2, size[1])),axis=0),axis=0)
            else:
                d1 = np.int16(np.floor((size[0]-resize[0])/2))
                T = Temp[d1:d1+resize[0], :, i]

            if size[1] < resize[1]:
                d1 = np.int16(np.floor((resize[1]-size[1])/2))
                d2 = np.int16(resize[1]-size[1]-d1)

                T = np.append(np.zeros((resize[0],d1)), np.append(T, np.zeros((resize[0],d2)),axis=1),axis=1)
            else:
                d1 = np.int16(np.floor((size[1]-resize[1])/2))
                T = T[:, d1:d1+resize[1]]


            frame = np.dstack((T[:,:], np.zeros(resize), np.zeros(resize)))

            frames.append(frame)

        return np.array(frames)
    def generateRandomData(self):
        angle = 360*np.random.random()
        Temp = self.Temp
        Temp = ndimage.rotate(Temp, angle=angle, reshape=False, mode='nearest')
        if np.random.random()>=0.5:
            Temp = np.fliplr(Temp)
        if np.random.random()>=0.5:
            Temp = np.flipud(Temp)
        Temp = Temp+1*(np.random.random(Temp.shape)-0.5)

        return Temp

    def saveTemp(self):
        Temp = self.Temp

        return Temp
    
