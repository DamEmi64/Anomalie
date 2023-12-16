import os
from turtle import color
import numpy as np
import pywt
import matplotlib.image  as img
import matplotlib.pyplot as plt

pattern_swap_change_halfway = './data/pattern_swap/change_halfway'
bitrate_fluctuation_change_halfway ='./data/bitrate_fluctuation/change_halfway'
sum_diff_change_halfway = './data/sum_diff/change_halfway'
pattern_swap_change_three_quarters = './data/pattern_swap/change_three_quarters'
bitrate_fluctuation_change_three_quarters ='./data/bitrate_fluctuation/change_three_quarters'
sum_diff_change_three_quarters = './data/sum_diff/change_three_quarters'

def readfile(folder_path):
    data = [] 
    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            with open(file_path, 'r') as infile:
              buf = [file_path,list(map(float,infile.read().splitlines()))]
              data.append(buf)
    return data    

def DWT(path,data_files):
    if not os.path.isdir('./images'+path):
        os.makedirs('./images'+path)
        
    i = 0
    for x, data in data_files:
        cA,cD = pywt.dwt(data, 'db1')
        plt.xlabel('sample')
        plt.ylabel('cD')
        plt.plot(cD)
        plt.title(str(i))
        plt.savefig(os.path.abspath('./images'+path+'/'+str(i)))
        plt.clf()
        i+=1

        
print('generate pattern_swap_change_halfway')        
DWT(pattern_swap_change_halfway,readfile(pattern_swap_change_halfway))
print('generate bitrate_fluctuation_change_halfway')     
DWT(bitrate_fluctuation_change_halfway,readfile(bitrate_fluctuation_change_halfway))
print('generate sum_diff_change_halfway')     
DWT(sum_diff_change_halfway,readfile(sum_diff_change_halfway))
print('generate pattern_swap_change_three_quarters')     
DWT(pattern_swap_change_three_quarters,readfile(pattern_swap_change_three_quarters))
print('generate bitrate_fluctuation_change_three_quarters')     
DWT(bitrate_fluctuation_change_three_quarters,readfile(bitrate_fluctuation_change_three_quarters))
print('generate sum_diff_change_three_quarters')     
DWT(sum_diff_change_three_quarters,readfile(sum_diff_change_three_quarters))