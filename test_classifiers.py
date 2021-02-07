time_range=15

import os
#import netCDF4
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

from sklearn.metrics import make_scorer

from joblib import Parallel, delayed
from sklearn.tree import *

from helpers.fourierExtrapolation import *
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

import dask
#dask.config.set(scheduler='processes')
from joblib import parallel_backend
from distributed import Client, progress, LocalCluster


def my_custom_loss_func(ground_truth, predictions):
    #diff = np.abs(ground_truth - predictions).max()
    #return np.log(1 + diff)
    return np.corrcoef(ground_truth,  predictions)[0,1] 

my_scorer = make_scorer(my_custom_loss_func, greater_is_better=True)
def run():
  global time_range
  #from helpers import helpers
  #from spark_sklearn import GridSearchCV
  for day in range(13, time_range):
    days = day
    print("------")
    print("day: " + str(days) )
    X = []
    y = []
    trends = []
    for fil in full_file_paths:
      #X, y = [], []
      X_temp = []
      y_temp = []
      count = 0
      with open(fil, "r") as f:
        #print(fil)
        lines = f.readlines()[1:]
        #if len(lines)>0:
        for line in lines:
          #print(line)
          val = line.strip().split(",")
          x = [float(val[a]) for a in range(len(val )- 1 ) ] #last one is date
          x.extend([float(a) for a in val[-1].split(" ")[0].split("/")  ])
          x.extend([ float(str(a)[0] + "." + str(a)[1]) for a in val[-1].split(" ")[1].split(":") ])
          y_ = [float(val[a]) for a in range(len(val ) ) if a==index ]
          #if np.nan not in x and np.nan not in y:
          X_temp.append(x)
          y_temp.append(y_[0] )
      
      x_val = X_temp
      y_val = y_temp
      #x_val = X_temp[:len(X_temp) - int(days*24*2)]
      #y_val = y_temp[int(days*24*2):]
      trend = [0 for a in y_val ]
      #print(len(x_val))
      #print(len(y_val))
      #print( len(y_val))
      if len(y_val) == 0: # file with no element
        continue
      x_val, y_val, trend = difference(x_val, y_val, int(days*24*2) )
      for a in range(0, len(x_val)): #remove first val coz trend=y[i+1]-y[i]. so no last val
          #print(x_val)     
          if not np.isnan(x_val[a]).any()  and not np.isnan(y_val[a]):# and y_val[a] < 20:
            #if all(a>-1000 for a in x_val[a]) and all(a<1000 for a in x_val[a]): 
            X.append(x_val[a])
            y.append(y_val[a])
            trends.append(trend[a])
      #X.extend(X_temp[:len(X_temp) - days*24*2])
      #y.extend([a[0]  for a in y_temp[days*24*2:] ] )

    #timators':[ 100, ], y = X[:50000], y[:50000]
    print(len(X))
    #print(trends[:10])
    trends = np.array(trends)
    indices = [a for a in range(len(y)) ]
    print("mean: " + str(np.mean(y)) + " std: " + str(np.std(y)) + " range: " + str([min(y), max(y) ]) )
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, indices, test_size=0.3, random_state=27)  
   
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    tr_trend = np.array(trends[idx1] )
    te_trend =np.array( trends[idx2] )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("train: " + str(len(X_train)) + ",  test: " + str(len(y_test) )  ) 
    grid1 = lg.LGBMRegressor(n_estimators=100, max_depth=None, random_state=100, n_jobs=-1)
    grid2 = ExtraTreesRegressor(random_state=100, max_depth=None, n_jobs=1)
    grid3 = DecisionTreeRegressor()
    grids = [('lgb', grid1),('et', grid2), ]#('dt', grid3) ]
    param2 = {'n_estimators':[ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],# 170, 220, 260,  350,  450,  550,  650, 700, 800 ],
      'min_samples_split':[2, 5, 10, 15, 25, 35, 45, 55, 65],
      'min_samples_leaf':[100, 150, 200, 250, 300, 350, 400, 450, 500],
      }
    param1 = { 'n_estimators':[ 100, 200, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], # 130, 200,  320, 370, 470, 540, 600, 700,  ],
      'num_leaves':[ 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, ], #5, 10, 15, 20, 30, 40, 50, 60, 70, 90, 120, 130],
      'min_child_samples':[ 5, 10, 20, 30, 50, 75, 100, 150, 200,],    
      }
    param3 = { 'n_estimators':[ 100, ],
      'num_leaves':[5, 10, 15, 20, 30, 40, 50, 60, 70],
      'min_child_samples':[ 5, 15, 25, 30,  35, 40, ],
    }

    c_range = np.linspace(-5,15,num=25)
    C_range = [math.pow(2,i) for i in c_range]
    #param3 = {}#{'alpha': C_range,  }
    
    params = [param1, param2, param3]  
    for gr in range(len(grids)):
      print("classifier: " + grids[gr][0])
      grid = grids[gr][1]
      #my_f = BlockingTimeSeriesSplit(n_splits=3)
      my_f = KFold(n_splits=3)
      X, y = np.array(X_train), np.array(y_train)
      y_ori = y.copy()
      param = {}
      #with parallel_backend('spark'):
      #param = gridsearch(X, y, grid, my_f, metric=r2_score, param_space= params[gr], n_jobs=-1 )
      #param = json.loads(param)
      grid = grid.set_params(**param)        
      grid = GridSearchCV( grid, params[gr], scoring='neg_mean_squared_error', n_jobs=-1, cv=3)
      #with parallel_backend('spark'):
      grid.fit(X, y )
      #print("best score: "+ str(grid.best_score_) + " best params: "+ str(grid.best_params_) )
      if grids[gr][0] != 'linear regression':
        param = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900],}# 'n_jobs':[-1]}
        print("best score: "+ str(grid.best_score_) + " best params: "+ str(grid.best_params_) )
        grid = grid.best_estimator_
        #if grids[gr][0] != 'et':
        #  param = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, ], }
        #  grid = GridSearchCV( grid, param, scoring='r2', n_jobs=1, cv=3 )  
        #else:
        #param = {'n_estimators': 500, 'n_jobs': -1}
        grid.set_params(**param)  
      #else:
      #  print("best score: "+ str(grid.best_score_) + " best params: "+ str(grid.best_params_) )
      predicted_1 = Parallel(n_jobs=-1)(delayed(get_test)(grid, X_train[train], y_train[train], X_train[test] ) for train,test in my_f.split(X_train, y_train) )          
      predicted_1 = [item for sublist in predicted_1 for item in sublist]
          
      y_selected = []
      y_ori = [y_train[a] +tr_trend[a] for a in range(len(y_train)) ]
      y_ori = np.array(y_ori)
      predict_ = []

      for tr, te in my_f.split(X_train, y_train):
        y_tes = y_ori[te]
        y_selected.extend(y_tes)
        predict_.extend([ predicted_1[a]+ tr_trend[a] for a in te ])
      y_ = y_selected
      predicted_1 = predict_
          
      #predicted_1 = cross_val_predict(grid, X, y, cv=5, n_jobs=-1)
      var = explained_variance_score(y_, predicted_1)
      abs_err = mean_absolute_error(y_, predicted_1)
      sq_err = mean_squared_error(y_, predicted_1)
      #r2 = r2_score(y, predicted_1)
      r2 = 1-(1-r2_score(y_, predicted_1))*((len(X_train)-1)/(len(X_train)-len(X_train[0])-1)) 
      mean_y, mean_pred = np.mean(y_), np.mean(predicted_1)
      #si = scatter_index(predicted_1, y) #observation put on end
      si = math.sqrt(np.mean([ ( -y_[a] + mean_y - mean_pred + predicted_1[a])**2 for a in range(len(y_)) ]) )/np.mean(y_) 
      nse = 1 - sum([(predicted_1[a] -y[a])**2 for a in range(len(y_))  ])/sum([ (y_[a]-mean_y)**2 for a in range(len(y_)) ])
      bias = np.mean([y_[a]-predicted_1[a] for a in range(len(y_))  ])
      hh = math.sqrt(sum([(y_[a]-predicted_1[a])**2 for a in range(len(y_))  ])/sum([y_[a]*predicted_1[a] for a in range(len(y_))  ]) )
      print("training")
      #print("mean: " + str(np.mean(y)) + " std: " + str(np.std(y)) + " range: " + str([min(y), max(y) ]) )
      print("sq_err: " + str(math.sqrt(sq_err)) + " abs_err: " + str(abs_err) + ' var: ' +
              str(var) + ' r2: ' + str(r2) + " si: " + str(si) + ' nse: ' + str(nse) +
              ' cc: ' + str(np.corrcoef(y_,  predicted_1)[0,1] ) + ' bias: ' + str(bias) +
              ' hh: ' + str(hh) )

      print("test")
      param= {'n_jobs':-1}
      grid.set_params(**param)
      #with parallel_backend('spark'):
      grid.fit(X_train, y_train)
      predicted_1 = grid.predict(X_test)
      y_t = [y_test[a] +te_trend[a] for a in range(len(y_test)) ]
      predicted_1 = [predicted_1[a]+ te_trend[a] for a in range(len(predicted_1)) ]
      var = explained_variance_score(y_t, predicted_1)    
      abs_err = mean_absolute_error(y_t, predicted_1)
      sq_err = mean_squared_error(y_t, predicted_1)
      #r2_score = r2_score(y_, predicted_1)
      r2 = 1-(1-r2_score(y_t, predicted_1))*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))
      mean_y, mean_pred = np.mean(y_t), np.mean(predicted_1)
      #si = math.sqrt(sq_err)/np.mean(predicted_1)
      #si = math.sqrt(sum([ ( -y_[a] + mean_y- mean_pred + predicted_1[a])**2 ]) /sum([b**2 for b in y_]) )
      si = math.sqrt(np.mean([ ( -y_t[a] + mean_y - mean_pred + predicted_1[a])**2 for a in range(len(y_t)) ] ) )/np.mean(y_t)
      nse = 1 - sum([(predicted_1[a] -y_t[a])**2 for a in range(len(y_t))  ])/sum([ (y[a]-mean_y)**2 for a in range(len(y_t)) ])
      bias = np.mean([y_t[a]-predicted_1[a] for a in range(len(y_t))  ])
      hh = math.sqrt(sum([(y_t[a]-predicted_1[a])**2 for a in range(len(y_t))  ])/sum([y_[a]*predicted_1[a] for a in range(len(y_t))  ]) )
      print("sq_err: " + str(math.sqrt(sq_err)) + " abs_err: " + str(abs_err) + ' var: ' +
              str(var) + ' r2: ' + str(r2) + " si: " + str(si) + ' nse: ' + str(nse) + 
              ' cc: ' + str(np.corrcoef(y_t, predicted_1)[0,1] )  + ' bias: ' + str(bias) +
              ' hh: ' + str(hh) )


if __name__ == "__main__":
  #with parallel_backend('spark'):
  run()


