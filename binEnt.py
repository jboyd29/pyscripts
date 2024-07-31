import os 
import struct
import numpy as np
import shutil
import csv

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from scipy.special import kv  #modified bessel function

hbarc = 0.197327

#iniFilepath = '../output/bin/1FC48DEF/1FC48DEF.ini'
#binFilepath = '../output/bin/1FC48DEF/temp.bin'



### This is all that needs pointed to the run directory
aHrunPath = './runs_20200611_1365/7/'



hsh = [d for d in os.listdir(aHrunPath+'output/bin') if os.path.isdir(os.path.join(aHrunPath+'output/bin', d))][0]

iniFilepath = aHrunPath + 'output/bin/' + hsh + '/' + hsh + '.ini'
binFilepath = aHrunPath + 'output/bin/' + hsh + '/temp.bin' 




def main():
    ini = configIni(filename = iniFilepath)
    ini.echoParams()
    
    

    d = readOBin(ini, binFilepath) # reads temperature binary file

    mDat = readMassData(ini, aHrunPath + ini['RealisticEquationOfStateDirectory'][2:]+'massData.tsv') #choses mass data based on ini
    intmD = interpMassD(ini, mDat) # interpolate mass data

    s = getEntropyFt(ini, d, intmD) #evaluate entropy inside freeze-out surface over evolution
    

    #write to file placed beside binary file in same directory
    headL = '#tau[fm/c]\t#S_tot[fm-1]'
    wrTar = aHrunPath+'output/bin/'+hsh+'/Stot.tsv'
    np.savetxt(wrTar, s, delimiter='\t', header=headL, fmt='%10.5e' )
    print('total entropy file written to: ', wrTar)
    
    
    #plot
    plt.plot(s[:,0], s[:,1])
    plt.yscale('log')
    plt.xlabel('tau [fm/c]')
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
    return np.array( np.array_split(np.array_split(np.array_split(dat, conf['TPts']*conf['XPts']*conf['YPts']), conf['TPts']*conf['XPts']), conf['TPts']) )

def fl(n):
    return int(np.floor(n))

def plotCentral(conf, bA):
    ti = np.arange(conf['TPts'])
    print(bA[ti, fl(conf['XPts']/2), fl(conf['YPts']/2), fl(conf['ZPts']/2)])
    plt.plot(ti ,bA[ti, fl(conf['XPts']/2), fl(conf['YPts']/2), fl(conf['ZPts']/2)])
    plt.xlabel('tau [fm/c]')
    plt.ylabel('T [fm-1]')
    plt.title('Central temp')
    plt.show()
        
    
def seq(m, T):
    return 4*np.pi*(1)*np.power(T,3)*np.power(m/T,2)*(4*kv(2,m/T)+(m/T)*kv(1,m/T))


def getEntropyFt(conf, bA, mI):
    tP = np.arange(conf['TPts'])
    xP = np.arange(conf['XPts'])
    yP = np.arange(conf['YPts'])
    zP = np.arange(conf['ZPts'])
    tPts = np.linspace(conf['TauMin'], conf['TauMin']+(conf['TauStepSize']*conf['RecordingFrequency']*conf['TPts']), conf['TPts'])
    sL = []
    for ti in tP:
        v = np.sum(np.heaviside(bA[ti].flatten()-conf['FinalTemperature'],0) * seq(mI(bA[ti].flatten()), bA[ti].flatten())) # heav(T>FinalTemperature) * seq
        sL.append([tPts[ti], v*conf['XSpacing']*conf['YSpacing']*conf['ZSpacing']]) # multiply by volume element size
    return np.array(sL) 

def readMassData(conf, fp):
    data = []
    with open(fp, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return np.array(data[2:]).astype(float)

def interpMassD(conf, mD):
    return interp1d(mD[:,0], mD[:,1], kind='cubic')






if __name__ == "__main__":
    main()
