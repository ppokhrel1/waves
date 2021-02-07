time_range=31
import pickle
import matplotlib.pyplot as plt
from helpers.fourierExtrapolation import *
import os
#import netCDF4
from sklearn.utils import parallel_backend
from pyspark import SparkContext, SparkConf
from joblibspark import register_spark

#sc = SparkContext(master='spark://nrl-05.cs.uno.edu:7077', appName='spark_features')

register_spark() # register spark backend

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
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef

from joblib import Parallel, delayed
from sklearn.tree import *

#from helpers.helpers import *
from sklearn.ensemble import *
from sklearn.linear_model import *
import json
import sys
from sklearn.preprocessing import StandardScaler
index = int(sys.argv[1])
#from spark_sklearn import GridSearchCV


from joblib import Parallel, delayed
def get_filepaths(directory):
	file_paths = []  # List which will store all of the full filepaths.
	# Walk the tree.
	for root, directories, files in os.walk(directory):
		for filename in files:
			# Join the two strings in order to form the full filepath.
			filepath = os.path.join(root, filename)
			file_paths.append(filepath  )  # Add it to the list.
	return file_paths  # Self-explanatory.
def fit_classifier(svc_rbf, param, metric, X_train, y_train, X_test, y_test):
	clas = svc_rbf.set_params(**param)
	clas.fit(X_train, y_train)
        #return param, roc_auc_score(y_test, [a[1]  for a in clas.predict_proba(X_test)] )
	return param, metric(y_test, clas.predict(X_test) )
def gridsearch(X, y, svc_rbf, skf, metric=roc_auc_score, param_space= {}, n_jobs=-1):
	import json
	my_hash = {}
	i=0
        #for train, test in skf.split(X, y):
	param_score = []
	#with parallel_backend('spark'):
	param_score = Parallel(n_jobs=n_jobs)(delayed(fit_classifier)(svc_rbf, param, metric, X[train], y[train], X[test], y[test] ) for param in ParameterGrid(param_space) for train, test in skf.split(X, y))
		#best_param, best_score = max(param_score, key=lambda x: x[1])
		#print('best param this generation is {} with score {}.'.format(best_param, best_score))
	for my_val in param_score:
            #key = [ (v, k) for k, v in my_val[0].iteritems ]
		key = json.dumps(my_val[0] )
		if key not in my_hash.keys() or my_hash[key]==None:
			my_hash[key ] = my_val[1]
		else:
			my_hash[key ] = my_hash[key ] + my_val[1]
        #best_param, best_score = max(param_score, key=lambda x: x[1])
        #print('Best scoring param is {} with score {}.'.format(best_param, best_score))
        #i+=1
	i = skf.get_n_splits()
	param_scores = []
	for k, v in my_hash.items():
		param_scores.append( (k, float(v/i) ) )
	best_param, best_score = max(param_scores, key=lambda x: x[1])
	print('Best scoring param is {} with score {}.'.format(best_param, best_score))
	return best_param

import glob
def get_test(clf, X, y, X_test):
	return [ a for a in clf.fit(X,y).predict(X_test) ]
def scatter_index(s, o):
	s_m = [a - np.mean(s) for a in s]
	o_m = [a-np.mean(o) for a in o]
	return math.sqrt(s_m-o_m)^2 / np.sum([a^2 for a in o])

full_file_paths = get_filepaths('./dataset/')

full_file_paths = full_file_paths[ : int(0.5* len(full_file_paths) ) ]

def difference(X, y,steps):
    x_val = X#[:len(X_temp) - steps]
    y_val = y#[a[0]  for a in y[steps: ]
    if steps == 0:
        return x_val, y_val, [0 for a in y_val]
    ret_val = [  ]
    trend = [  ]
    #print(y[:5])
    #ret_val = [a-b for a, b in zip(y, y[steps:])]
    #print(y[:10])
    ret_val = [y[a]-y[a-steps] for a in range(steps, len(y)) ]
    #ret_val.extend([y[a+steps]-y[a] for a in range(0, len(y) - steps)  ] )
    y_val = ret_val#[steps:]
    #print(len(X))
    #print(len(ret_val))
    #print(len(y))
    trend.extend([ y[a-steps] for a in range(steps, len(y))])
    return x_val[:len(x_val)-steps], ret_val, trend


import dask
#dask.config.set(scheduler='processes')
from joblib import parallel_backend
from distributed import Client, progress, LocalCluster

#client = Client("137.30.125.208:8786") 
#client = Client(port = 8999, processes=True)
def run():
        global time_range
	#from helpers import helpers
	#from spark_sklearn import GridSearchCV
        for day in range(1, time_range):
                days = day
                print("------")
                print("day: " + str(days) )
                X = []
                y = []
                trends = []
                for fil in full_file_paths:
                        X_temp = []
                        y_temp = []
                        with open(fil, "r") as f:
				#print(fil)
                                lines = f.readlines()[1:]
				#if len(lines)>0:
                                for line in lines:
					#print(line)
                                        val = line.strip().split(",")
                                        #x = [float(val[a]) for a in range(len(val ) ) ]# if a !=index ]
                                        x = [float(val[a]) for a in range(len(val )- 1 ) ] #last one is date
                                        x.extend([float(a) for a in val[-1].split(" ")[0].split("/")  ][1:] )
                                        x.extend([ float(str(a)[0] + "." + str(a)[1]) for a in val[-1].split(" ")[1].split(":") ])
                                        #y_ = [float(val[a]) for a in range(len(val ) ) if a==index ]
                                        y_ = [float(val[a]) for a in range(len(val ) ) if a==index ]
					#if np.nan not in x and np.nan not in y:
                                        #for v in range(len(x)):
                                        if y_[0] < 3000:
                                            X_temp.append(x[1:] )
                                            y_temp.append(y_)
                                        #for a in y:
                                        #if a > 3000:
                                        #print(fil)
                        #x_val = X_temp[:len(X_temp) - int(days*24*2)]
                        #y_val = [a[0]  for a in y_temp[int(days*24*2):] ]
                        x_val, y_val = X_temp, [a[0] for a in y_temp]
                        x_val, y_val, trend = difference(x_val, y_val, int(days*24*2) )		   
			#x_val = np.array(x_val)	
			#x_val = fourierExtrapolation(x_val, 24*2*1, 10)
                        for a in range(len(x_val)):
                            if not np.isnan(x_val[a]).any()  and not np.isnan(y_val[a]):
                                #X.append(x_val[a] )
                                #y.append(y_val[a])
                                X.append(x_val[a])
                                y.append(y_val[a])
                                trends.append(trend[a])
			#X.extend(X_temp[:len(X_temp) - days*24*2])
			#y.extend([a[0]  for a in y_temp[days*24*2:] ] )

		#timators':[ 100, ], y = X[:50000], y[:50000]
                
                print(len(X))
                print( len(X[0]) )
		#X = fourierExtrapolation(X, 24*2*1, 10)
                #y = difference(y)
                print("mean: " + str(np.mean(y)) + " std: " + str(np.std(y)) + " range: " + str([min(y), max(y) ]) )
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)	
		
                scaler = StandardScaler()
		
                X_train = scaler.fit_transform(X_train)
		
                X_test = scaler.transform(X_test)
		
                print("train: " + str(len(X_train)) + ",  test: " + str(len(y_test) )	) 
                
                grid2 = lg.LGBMRegressor(n_estimators=500, max_depth=None, num_leaves=5000,random_state=100, n_jobs=-1)
		
                grid1 = ExtraTreesRegressor(random_state=100, min_samples_leaf = 2, n_estimators=500, max_depth=None, n_jobs=-1)
		
                #grid3 = DecisionTreeRegressor()
                with parallel_backend('spark'):
                    grid2.fit(X_train, y_train)
                print("lg done")
                with parallel_backend('spark'):
                    grid1.fit(X_train, y_train)
                print("et done")
                importances = grid1.feature_importances_
                importances = importances#[: len(importances) - 3]
                std = np.std([tree.feature_importances_ for tree in grid1.estimators_], axis=0)
                indices = [a for a in range(len(importances) ) ]
                #print(importances)
                #plt.ylabel("Value"")
                #plt.xlabel("Features")
                #plt.plot(indices, lgb_train, label = "LightGBM-Train") 
                #plt.plot(indices, et_train, label = "ET-Train")
#plt.plot(indices, lr_train, label = "LR")

#plt.title(filename)
#plt.legend()
#plt.savefig('plots/'+filename+'_train.png')
#plt.close()
#import matplotlib.pyplot as plt

                #plt.bar(indices, lgb_test, label = "Feature Importance")
                #plt.plot(indices, et_test, label = "ET-Test")
#plt.plot(indices, lr_test, label = "LR")

                #plt.title(out_file)
                #X_train = np.array([ a[:len(a)-4] for a in X_train] )
                #importances = importances[:len(importances)-4]
                #indices = [a for a in range(len(importances) ) ]
                #indices_vals = [ "depth", "spread", "Hm0", "Tp", "Ta", "Tz", "psd", "Dp", "skew", "kurt", ]#"year", "month", "day" "time" ]
                #plt.title("Feature importance for Significant Wave Height")
                #plt.bar(range(X_train.shape[1] ), importances[indices], color="r", yerr=std[indices], align="center")
                #plt.xticks(range(X_train.shape[1] ), indices_vals)
                #plt.xlim([-1, X_train.shape[1] ])
                #plt.legend()
                #plt.savefig('feature_importance/'+'hs'+'.png')

                #plt.close()
                classifiers = {'et': grid1, 'lg': grid2}
                with open("classifiers/day" + str(day) + ".pkl", "wb") as f:
                    pickle.dump(classifiers, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	#import distributed.joblib  # noqa
	#from distributed import Client, Worker, Nanny, wait
	#from dask.distributed import Client, SSHCluster
	#from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
	#from ipyparallel import Client
	#from ipyparallel.joblib import IPythonParallelBackend
	
	#c = Client(profile='ssh')
	#bview = c.load_balanced_view()
	#register_parallel_backend('ipyparallel', lambda : IPythonParallelBackend(view=bview))
	
	#register_joblib_backend(name='ipyparallel', make_default=False)
	#cluster = LocalCluster(processes=False)
	#cluster = SSHCluster(
	#["localhost", "localhost" ],
	#connect_options={"known_hosts": None},
	#worker_options={"nthreads": 10},
	#)
	#client = Client(cluster 
	#	threads_per_worker=4,
	#	memory_limit='64GB',
	#	processes=False,
	#)
	#from pyspark import SparkContext, SparkConf
	#import findspark
	#findspark.init()
	#from joblibspark.backend import *
	#from joblibspark import register_spark
	#from joblibspark import *	
	
	#sc = SparkContext(master='spark://137.30.125.205:7077', appName='spark_features')
	#sc.addPyFile('helpers/helpers.py')	
	#findspark.init()
	#register_spark()
	#with parallel_backend('spark'):
	run()
	#client = Client("137.30.125.208:8786")	


