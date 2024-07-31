import os 
import struct
import numpy as np
import shutil
import csv

import matplotlib.pyplot as plt

from scipy.special import kv  #modified bessel function

hbarc = 0.197327

iniFilepath = '../output/bin/1FC48DEF/1FC48DEF.ini'
binFilepath = '../output/bin/1FC48DEF/temp.bin'

# ???????
mass = 10




def main():
    ini = configIni(filename = iniFilepath)
    ini.echoParams()
    
    

    d = readOBin(ini, binFilepath)

    

    s = getEntropyFt(ini, d, mass)
    plt.plot(s[:,0], s[:,1])
    plt.yscale('log')
    plt.xlabel('[fm/c]')
    plt.ylabel('S_tot')
    plt.title('Unnormalized entropy inside freeze-out surface')
    plt.show()








def parse_string(input_string):
    try:
        # Try to parse as an int 
        result = int(input_string)
    except ValueError:
        try:
            # try parsing as a float
            result = float(input_string)
        except ValueError:
            # keep it as a string
            result = input_string
    return result

class configIni: # for grabbing variables from ini file
    def __init__(self, filename="../params"):
        self.data = {}
        self.filename = filename
        self.grabParams()
    def __getitem__(self,key):
        return self.data[key]
    def __setitem__(self,key,value):
        self.data[key] = value
    def __len__(self):
        return len(self.data)
    def __str__(self):
        return str(self.data)
    def __iter__(self):
        return iter(self.data)
    def keys(self):
        return self.data.keys()
    def grabParams(self):
        try:
            with open(self.filename, 'r') as file:
                for line in file:
                    if (not line.startswith('#')) or (not line.startswith('[')):
                        items = line.strip().split()
                        if len(items) == 3:
                            key, non, val = items
                            self.data[key] = parse_string(val)
        except FileNotFoundError:
            print(f"File '{self.filename}' not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    def echoParams(self):
        for key in self.data.keys():
            print(key,':',self.data[key])


def readOBin(conf, binPath):
    with open(binPath, 'rb') as file:
        binary_data = file.read()
    num_floats = len(binary_data) // 4
    dat = np.array(struct.unpack(f'{num_floats}f', binary_data))
    conf['TPts'] = len(dat) // (conf['XPts']*conf['YPts']*conf['ZPts'])
    return hbarc * np.array( np.array_split(np.array_split(np.array_split(dat, conf['TPts']*conf['XPts']*conf['YPts']), conf['TPts']*conf['XPts']), conf['TPts']) )

def fl(n):
    return int(np.floor(n))

def plotCentral(conf, bA):
    ti = np.arange(conf['TPts'])
    print(bA[ti, fl(conf['XPts']/2), fl(conf['YPts']/2), fl(conf['ZPts']/2)])
    plt.plot(ti ,bA[ti, fl(conf['XPts']/2), fl(conf['YPts']/2), fl(conf['ZPts']/2)])
    plt.show()
        
    
def seq(m, T):
    return 4*np.pi*(1)*np.power(T,3)*np.power(m/T,2)*(4*kv(2,m/T)+(m/T)*kv(1,m/T))


def getEntropyFt(conf, bA, m):
    tP = np.arange(conf['TPts'])
    xP = np.arange(conf['XPts'])
    yP = np.arange(conf['YPts'])
    zP = np.arange(conf['ZPts'])
    tPts = np.linspace(conf['TauMin'], conf['TauMin']+(conf['TauStepSize']*conf['RecordingFrequency']*conf['TPts']), conf['TPts'])
    sL = []
    for ti in tP:
        v = np.sum(np.heaviside(bA[ti].flatten()-conf['FinalTemperature'],0) * seq(m, bA[ti].flatten())) # heav(T>FinalTemperature) * seq
        sL.append([tPts[ti], v*conf['XSpacing']*conf['YSpacing']*conf['ZSpacing']]) # multiply by volume element size
    return np.array(sL)


if __name__ == "__main__":
    main()
