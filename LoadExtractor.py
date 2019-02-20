####Importing the necessary libraries####
import os
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
import python_speech_features as speech_features

class LoadExtractor:
    
    def __init__(self, path, nfft=610):
        '''
        path: path to the folder containing the dataset from TESS
        nfft: the FFT size, default is 610
        '''
        self.path=path
        self.nfft=nfft
        
    def loader(self):
        label = 0
        sigList = []
        rateList = []
        labelList = []
        classList = os.listdir(self.path)
        switchLen = len(classList)/2
        for nextDir in tqdm(os.listdir(self.path)):
            classFolder = os.path.join(self.path, nextDir)
            for file in os.listdir(classFolder):
                if file is not None:
                    try:
                        (rate, sig) = wav.read(os.path.join(classFolder, file))
                        rateList.append(rate)
                        sigList.append(sig)
                        labelList.append(label)
                    except ValueError:
                        print(os.path.join(classFolder, file)+" is corrupted")
                        pass
            label += 1
            if label == switchLen:
                label = 0
        return rateList, sigList, labelList
    
    def mfccExtractor(self, sigList, rateList):
        '''
        sigList: list of audio signals for which we need to obtain the MFCC features
        rateList: list of sampling rates of the respective signals
        '''
        mfccFeatureList = []
        for audio in zip(sigList, rateList):
            audioSig = audio[0]
            audioRate = audio[1]
            mfccFeatures = speech_features.mfcc(audioSig, audioRate, nfft = self.nfft)
            mfccFeatureList.append(mfccFeatures)
        return mfccFeatureList
    
    def deltaFeatureExtractor(self, featureList, interval=2):
        '''
        featureList: the list of feature vectors for which we are to obatin the delta feature vector
        interval: the number of frames with respect to which the delta value is calculated, default is 2
        '''
        deltafeatureList = []
        for feature in tqdm(featureList):
            feature = np.asarray(feature)
            deltafeatureList.append(speech_features.delta(feature, interval))
        return deltafeatureList
    
    def get_featureData(self):
        rateList, sigList, labelList = self.loader()
        mfccFeatureList = self.mfccExtractor(sigList, rateList)
        deltaList = self.deltaFeatureExtractor(mfccFeatureList)
        deltaDeltaList = self.deltaFeatureExtractor(deltaList)
        
        featureList = []
        for features in zip(mfccFeatureList, deltaList, deltaDeltaList):
            features = np.concatenate((features[0], features[1], features[2]),axis=1)
            featureList.append(features)
        return featureList
