## script that saves file

import sys

import joblib
#from ray.util.joblib import register_ray
#register_ray()

day_ahead = int(sys.argv[1])
stations = [ '149', '150', '151', '152', '155', '158', '161', '165', 
'169', '172', '175', '181', '186', '189', '190', 
'191', '201', '203', '207', '208', '209', '210',
'211', '212', '213', '214', '215', '216', '217',
#'218', '219', '210', '211', '212', '213', '214',
#'215', '216', '217', '218', '219', '210',
]
#stations = ['149']
deploy = ['01', '02'] # Set deployment number from .nc file

print("----------------------")
print("days: " + str(day_ahead) )
time_length = 2*24*day_ahead
#startdate = "05/21/2018 09:00" # MM/DD/YYYY HH:MM
#enddate = "05/21/2018 10:00" # MM/DD/YYYY HH:MM
#duration  = 30 # Set length of timeseries (minutes)
#qc_level = 2 # Filter data with qc flags above this number 

import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import calendar
from helpers.time_ranges import * 
from second_order import *
import math
import lightgbm as lg
from sklearn.model_selection import *
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance, mean_gamma_deviance
from joblib import Parallel, delayed
# CDIP Archived Dataset URL

def zero_cross(a, b, c):
        if (a < 0 and b > 0 ):
                return True
        return False
def get_crest_height(array):
        zero_c = [a for a in range(1, len(array)-1) if zero_cross(array[a-1], array[a], array[a+1]) ]
        heights = [array[zero_c[a]: zero_c[a+ 1] ] for a in range(0, len(zero_c) - 1)]
        heights_norm = np.max( [max(a)-min(a) for a in heights] )
        return heights_norm

def get_data(stn, deploy):
        data_url = './thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/' + stn + 'p1/' +  deploy

        nc = netCDF4.Dataset(data_url)
        nc.set_auto_mask(False)

        ncTime = nc.variables['waveTime'][:]
        Dmean = nc.variables['waveMeanDirection']
        Fq = nc.variables['waveFrequency'] #size is 64
        Ed = nc.variables['waveEnergyDensity']
        Hs = nc.variables['waveHs']
        
        #print(nc.variables )
        Tp = nc.variables['waveTp']
        Dp = nc.variables['waveDp']
        Ta = nc.variables['waveTa']
        Psd = nc.variables['wavePeakPSD']
        bandwidth = nc.variables['waveBandwidth']

        flags_for_30_min = nc.variables['waveFlagPrimary']

        #print(len([a for a in flags_for_30_min if a==1 ] ))
        #print([a for a in flags_for_30_min ] )
        #print(len([a for a in flags_for_30_min if str(a)=='1' ]) )
        wave_flags = nc.variables['waveFlagPrimary']
        #print(len([a for a in wave_flags]) )
        #print(nc.variables) 
        #print(len(  nc.variables['waveA1Value'] ))
        a1 = nc.variables['waveA1Value']
        a2 = nc.variables['waveA2Value']
        b1 = nc.variables['waveB1Value']
        b2 = nc.variables['waveB2Value']

        zero_upcross = nc.variables['waveTz']

        depth = nc.variables['metaWaterDepth']
        
        spread = nc.variables['waveSpread']
        spread = [np.mean(a) for a in spread]
#print(a1[0])
        zdisp = nc.variables['xyzZDisplacement']
        zdisp = np.ma.MaskedArray(zdisp )
        #interval=2305 ##1.2*30*60
        #interval_vals = [zdisp[a*interval: (a+1)*interval] for a in range(0, math.floor(len(zdisp)/interval )  )]
        #print(len(interval_vals))
        #print(len(a1))
        #heights = [ get_crest_height(a) for a in interval_vals if len(a)>0] 

        unixstart = getUnixTimestamp(startdate,"%m/%d/%Y %H:%M") 
        neareststart = find_nearest(ncTime, unixstart)  # Find the closest unix timestamp
        nearIndex = np.where(ncTime==neareststart)[0][0]  # Grab the index number of found date

        Index = nearIndex-16 # Subtract 16 index units from the waveTime variable, because it appears to be shifted forward 16 from UTC time. 
                     # NOTE: CDIP plots are displayed in PST (UTC-8 hrs).

        EndDate = getHumanTimestamp(Index,"%m/%d/%Y %H:%M")
        #print(EndDate)

        unixend = getUnixTimestamp(enddate,"%m/%d/%Y %H:%M")
        future = find_nearest(ncTime, unixend)  # Find the closest unix timestamp
        futureIndex = np.where(ncTime==future)[0][0]  # Grab the index number of found date

        EndDate = getHumanTimestamp(futureIndex,"%m/%d/%Y %H:%M")
        #dates = [ getHumanTimestamp(a,"%m/%d/%Y %H:%M") for a in ncTime ]

#now, lets get frequency, direction, a1, a2, b1, b2 
# a_vals = []
# for m in range(len(a1)):
#       v_ = []
#       for val in range(len(a1[m] ) ):
#               v_.extend([ a1[m][val], a2[m][val]  ])
#       a_vals.append(v_)

#print("apple")
#print(Fq[1])
        size = len(Hs)
        dates = [ getHumanTimestamp(a,"%m/%d/%Y %H:%M") for a in ncTime ]#print("running classifiers")

        if size < 2000:
                return [], []
        a1 = np.ma.MaskedArray(a1[:size])
        a2 = np.ma.MaskedArray(a2[:size])
        b1 = np.ma.MaskedArray(b1[:size])
        b2 = np.ma.MaskedArray(b2[:size])
        Dmean = np.ma.MaskedArray(Dmean[:size])
        Fq = np.ma.MaskedArray(Fq[:size] )
        depth = np.ma.MaskedArray(depth[:size] )
        bandwidth = np.ma.MaskedArray(bandwidth[:size] )
        a1_norm = []
        a2_norm = []
        b1_norm = []
        b2_norm = []
        energy_density=[]
        #print(bandwidth)
        for a in range(len(Ed)):
                ed = 0
                a1_, a2_, b1_, b2_ = 0, 0, 0, 0
                for b in range(len(Ed[a] )):
                        #print(bandwidth[a])
                        ed += bandwidth[b] * Ed[a][b]
                        a1_ += bandwidth[b] * Ed[a][b] * a1[a][b]
                        a2_ += bandwidth[b] * Ed[a][b] * a2[a][b]
                        b1_ += bandwidth[b] * Ed[a][b] * b1[a][b]
                        b2_ += bandwidth[b] * Ed[a][b] * b2[a][b]
                energy_density.append(ed) 
                a1_norm.append( a1_ /ed )
                a2_norm.append( a2_ /ed )
                b1_norm.append( b1_ /ed )
                b2_norm.append( b2_ /ed )
        #print(len(Ed))
        #print(len(Ed[0]))                                      
        angles = [math.atan(b1_norm[a]/a1_norm[a]) for a in range(size)  ]
        m1 = [( (a1_norm[a])**2 + (b1_norm[a])**2)**(1/2) for a in range(size) ]
        m2 = [a2_norm[a]*math.cos(math.radians(2*angles[a])) + b2_norm[a]*math.sin(math.radians(2*angles[a])) for a in range(size) ]
        n2 = [b2_norm[a]*math.cos(math.radians(2*angles[a])) - a2_norm[a]*math.sin(math.radians(2*angles[a])) for a in range(size) ]
        
        skew = [-n2[a]/( ( (1-m2[a])/2)**(3/2) ) for a in range(size) ]
        kurt = [(6-8*m1[a]+2*m2[a])/( (2*(1-m1[a]))**2) for a in range(size) ]
        
        #print(depth)

        arr= [[ dates[a]] for a in range(size) ]
#arr = arr_1
        arr = [np.concatenate(( [depth], [spread[a]], [ Hs[a] ], [zero_upcross[a] ], [ Tp[a] ], [Psd[a] ], [ Ta[a] , Dp[a] ], ), axis=0) for a in range(size) ]
        arr = [np.concatenate( (arr[a], [skew[a]], [kurt[a]]  )) for a in range(size) ]

        #arr = [ arr[a] for a in range(len(arr)) if str(flags_for_30_min[a])=='1' ]

        y_vals = spread


        ##doing the data first
        #X = [ arr[a] for a in range(len(arr[:size - time_length] ) )  ]
        #y = [a for a in y_vals[time_length: size] ]

        X = [a for a in arr ]
        y = [a for a in y_vals]
        if len(X) < time_length:
                return [], []
        #all_vals = [np.concatenate(( [y[a] ], X[a]  ) ) for a in range(len(X) ) ]
        #print(X[0] )
        fg = list(flags_for_30_min).copy()
        #for a in range(time_length, len(flags_for_30_min) ):
        #        if str(flags_for_30_min[a - time_length ])=="1" :       #earlier was just for y, this will take care if flag is in features (X)  side
        #                fg[a] = "1"
        flags_for_30_min = fg
        print( str(len(flags_for_30_min)) + "  " + str(len(arr)) )
        X = [np.concatenate( ( [flags_for_30_min[a]] , X[a], [ dates[a] ] )  ) for a in range(len(X)) ]
        #X = [ all_vals[a][:] for a in range(len(all_vals)) ]#[::4] #every 4th element
        #y = [ flags_for_30_min[a] for a in range(len(all_vals)) ]#[::4] #every 4th element

        return X, y

def get_test(clf, X, y, X_test):
        return [ a for a in clf.fit(X,y).predict(X_test) ]
def scatter_index(s, o):
        s_m = [a - np.mean(s) for a in s]
        o_m = [a-np.mean(o) for a in o]
        return math.sqrt(s_m-o_m)^2 / np.sum([a^2 for a in o])


if __name__ == "__main__":
        from helpers.helpers import *
        from joblib import Parallel, delayed
        mp = []
        for stn in stations:
                #print("station: " + str(stn) )
                paths = [ fil for fil in os.listdir('./thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/' + stn + 'p1/') if not fil.endswith("historic.nc") ]
                paths = sorted(paths)
                for a in paths:
                        mp.append([stn, a ])
                        
        print(mp)
        vals = Parallel(n_jobs=-1)(delayed(get_data)(val[0], val[1]) for val in mp)
        #print(X_val)
        for a in range(len(vals)):
                with open("dataset/"+ str(a) + ".txt", "w") as f:
                        f.write("flag, time, depth, spread, hs, tz, tp, psd, ta, Dmean, skew, kurt, date \n")
                        for lines in vals[a][0]:
                                f.write(",".join(str(a) for a in lines)+"\n")   
        X, y = [], []
        for a in vals:
                X.extend(a[0] )
                y.extend(a[1] )
        print(len(X))
        X = np.array(X)
        y = np.array(y)
#print(X[0] )
        grid = lg.LGBMRegressor(n_estimators=100, max_depth=None, random_state=100, n_jobs=-1)
        
        my_f = BlockingTimeSeriesSplit(n_splits=5)
        predicted_1 = Parallel(n_jobs=-1)(delayed(get_test)(grid, X[train], y[train], X[test]) for train,test in my_f.split(X, y) )
        predicted_1 = [item for sublist in predicted_1 for item in sublist]
        y_selected = []
        for tr, te in my_f.split(X, y):
                y_tes = y[te]
                y_selected.extend(y_tes)
        y = y_selected
        #predicted_1 = cross_val_predict(grid, X, y, cv=5, n_jobs=-1)
        var = explained_variance_score(y, predicted_1)
        abs_err = mean_absolute_error(y, predicted_1)
        sq_err = mean_squared_error(y, predicted_1)
        r2 = r2_score(y, predicted_1)

        #si = scatter_index(predicted_1, y) #observation put on end
        si = math.sqrt(sq_err)/np.mean(y)
        print("mean: " + str(np.mean(y)) + " std: " + str(np.std(y)) + " range: " + str([min(y), max(y) ]) ) 
        print("sq_err: " + str(math.sqrt(sq_err)) + " abs err: " + str(abs_err) + ' var :' +
                 str(var) + ' r2: ' + str(r2) + " si: " + str(si) )



