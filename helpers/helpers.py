#from imblearn.under_sampling import *
from pprint import pprint
import numpy as np
import pandas as pd
from collections import defaultdict
import copy

import json
#from sklearn.externals import joblib
import joblib
from sklearn.model_selection import * #ShuffleSplit
#from sklearn.cross_validation import *
import os
from sklearn.ensemble import *

from sklearn.datasets import load_digits
from pprint import pprint
from subprocess import call
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef
#from subprocess import call
from sklearn.model_selection import *

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils.validation import check_non_negative, _deprecate_positional_args
import warnings
import numbers
import time
from traceback import format_exc
from contextlib import suppress

import scipy.sparse as sp
from joblib import Parallel, delayed

from sklearn.base import is_classifier, clone
from sklearn.utils import (indexable, check_random_state, _safe_indexing,
                     _message_with_time)
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder
import multiprocessing
from multiprocessing import Pool
#step = sys.argv[1]
#window_ = sys.argv[2]
#paths = os.listdir("buoy_data/test_step" + step +"/window_"+ window_ +"_dataset")
#random.shuffle(paths)
#sc = SparkContext(master='spark://nrl-05.cs.uno.edu:7077', appName='spark_features')

#register_spark() # register spark backend
#register()
chromosomes = "1000100010101010001000000100000000111001100010010100001010110000010001110111110000101000011000001000101000000011111000111111111"
chromosomes = "1000100010000001010000000111000001000000000000010000001010100100001011000000011110011100011001001100000100001110011110101111100001010110101001000011110101000011111111"
#def fit_classifier(klass, param, metric, X_train, Y_train, X_validation, Y_validation):
#    classifier = klass.fit(X_train, Y_train)
#    Y_predict = classifier.predict(X_validation)
#    score = metric(Y_validation, Y_predict)
#    return (param, score)

def fit_classifier(svc_rbf, param, metric, X_train, y_train, X_test, y_test):
	clas = svc_rbf.set_params(**param)
	clas.fit(X_train, y_train)
	#return param, roc_auc_score(y_test, [a[1]  for a in clas.predict_proba(X_test)] )
	return param, metric(y_test, clas.predict(X_test) )
#res=BalanceCascade(random_state=0)


skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

def gridsearch(X, y, svc_rbf, skf, metric=roc_auc_score, param_space= {}, n_jobs=-1):
	import json
	my_hash = {}
	i=0
	#for train, test in skf.split(X, y):
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

def get_filepaths(directory):
        file_paths = []
        for root, directories, files in os.walk(directory):
                for filename in files:
                        # Join the two strings in order to form the full filepath.
                        filepath = os.path.join(root, filename)
                        file_paths.append(filepath  )  # Add it to the list.
        return file_paths  # Self-explanatory.

def print_scores(y, predicted):
    print( roc_auc_score(y, predicted) )
    print( 1 - roc_auc_score(y, predicted) )
    predicted = [ a[0]>0.5 for a in predicted ]
    confusion = confusion_matrix(y, predicted)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    #Specificity
    SPE_cla = (TN/float(TN+FP))

    #False Positive Rate
    FPR = (FP/float(TN+FP))

    #False Negative Rate (Miss Rate)
    FNR = (FN/float(FN+TP))

    #Balanced Accuracy
    ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))
    #compute MCC
    MCC_cla = matthews_corrcoef(y, predicted)
    F1_cla = f1_score(y, predicted)
    PREC_cla = precision_score(y, predicted)
    REC_cla = recall_score(y, predicted)
    Accuracy_cla = accuracy_score(y, predicted)
    print('TP = ', TP)
    print('TN = ', TN)
    print('FP = ', FP)
    print('FN = ', FN)
    print('Recall/Sensitivity = %.5f' %REC_cla)
    print('Specificity = %.5f' %SPE_cla)
    print('Accuracy_Balanced = %.5f' %ACC_Bal)
    print('Overall_Accuracy = %.5f' %Accuracy_cla)
    print('FPR_bag = %.5f' %FPR)
    print('FNR_bag = %.5f' %FNR)
    print('Precision = %.5f' %PREC_cla)
    print('F1 = %.5f' % F1_cla)
    print('MCC = %.5f' % MCC_cla)



class Nystroem(TransformerMixin, BaseEstimator):
	def __init__(self, kernel="rbf", *, gamma=None, coef0=None, degree=None,
		kernel_params=None, n_components=100, random_state=None):
		self.kernel = kernel
		self.gamma = gamma
		self.coef0 = coef0
		self.degree = degree
		self.kernel_params = kernel_params
		self.n_components = n_components
		self.random_state = random_state
	def fit(self, X, y=None):
		#X = self._validate_data(X, accept_sparse='csr')
		rnd = check_random_state(self.random_state)
		n_samples = X.shape[0]
		if self.n_components > n_samples:
			n_components = n_samples
			warnings.warn("n_components > n_samples. This is not possible.\n"
				"n_components was set to n_samples, which results"
				" in inefficient evaluation of the full kernel.")

		else:
			n_components = self.n_components
		#n_components = min(n_samples, n_components)
		#n_folds = max( int( n_samples / n_components), 2)
		#rus = RandomUnderSampler(random_state=42)
		#X_res, y_res = rus.fit_resample(X, y)
		#X_res, y_res = X, y
		n_components = min(n_samples, n_components)
		n_folds = max( int( len(X_res) / n_components), 2)
		skf = StratifiedKFold(n_splits = n_folds)
		_, indices = [a for a in skf.split(X_res, y_res) ] [0]
		basis = X_res[indices]
		basis_kernel = pairwise_kernels(basis, metric=self.kernel,
			filter_params=True,
			**self._get_kernel_params())
		U, S, V = svd(basis_kernel)
		S = np.maximum(S, 1e-12)
		self.normalization_ = np.dot(U / np.sqrt(S), V)
		self.components_ = basis
		self.component_indices_ = indices
		return self

	def transform(self, X):
		check_is_fitted(self)
        
		X = check_array(X, accept_sparse='csr')
		kernel_params = self._get_kernel_params()
		embedded = pairwise_kernels(X, self.components_,
			metric=self.kernel,
			filter_params=True,
			**kernel_params)
		return np.dot(embedded, self.normalization_.T)


	def _get_kernel_params(self):
		params = self.kernel_params
		if params is None:
			params = {}
		if not callable(self.kernel) and self.kernel != 'precomputed':
			for param in (KERNEL_PARAMS[self.kernel]):
				if getattr(self, param) is not None:
					params[param] = getattr(self, param)
		else:
			if (self.gamma is not None or
				self.coef0 is not None or
				self.degree is not None):
				raise ValueError("Don't pass gamma, coef0 or degree to "
					"Nystroem if using a callable "
					"or precomputed kernel")
		return params
	

from joblib import Parallel, delayed

def fit(clf, X_train, y_train, X_test, test, clf_ind):
	#clf.fit(X_test, y_test)
	return [clf_ind, test, [ a[0] for a in clf.fit(X_train, y_train).predict_proba(X_test) ]  ]

def train(clf, X, y, send_end):
	clf = copy.deepcopy(clf)
	clf.fit(X, y)
	send_end.send(clf ) 
def train_base(classifiers, X, y, kfold):
	all_probs = []
	for a in range(len(X) ):
		all_probs.append([0 for a in classifiers  ] )
	#all_probs = np.array(all_probs)
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	jobs = []	
	pipe_list = []
	trained_cls = [pool.apply_async(cls.fit, args=(X, y) ) for cls in classifiers ]
	
	for train, test in kfold.split(X, y):
		X_train, y_train = X[train], y[train]
		X_test, y_test = X[test], y[test]
		
		for a in range(len(classifiers)):
			cls = classifiers[a]
			p = pool.apply_async(fit, args=(cls, X_train, y_train, X_test, test, a),  )
			jobs.append(p)
		'''
		probs = Parallel(n_jobs=-1) ( delayed(fit) (clf, X_train, y_train, X_test, y_test ) for clf in classifiers)	
		probs = np.array(probs).T
		probs = list(probs)
		#print(probs)
		i = 0
		for a in test:
			all_probs[a] = probs[i]
			i+=1	
		#np.put(all_probs, test, probs)
		#print(all_probs)	
		'''
	#train and return classifiers
	#clf_jobs = []
	#recv_end, send_end = multiprocessing.Pipe(False)
	#for clf in classifiers:
	#	p = multiprocessing.Process(target=train, args=(clf, list(X), list(y), send_end ) )
	#	p.start()
	#	clf_jobs.append(p)
		
	#trained_cls = [a.get()[0] for a in clf_jobs ]
	trained_cls = [a.get() for a in trained_cls ]	
	result_list = [a.get() for a in jobs ]
	
	pool.close()
	pool.join()
	return_val = [[0]* len(classifiers)  for b in X]
	
	for a in result_list:
		cls = a[0]
		indices = a[1]
		values = a[2]
		i = 0
		for b in indices:
			return_val[b][cls] = values[i]
			i+=1		
	#return np.array(all_probs)
	return_val = [a for a in return_val if list(set(a) )!=[0] ] #remove empty elements
	return np.array(return_val), trained_cls




class BlockingTimeSeriesSplit:
	def __init__(self, n_splits):
		self.n_splits = n_splits
	def get_n_splits(self, X, y, groups):
		return self.n_splits
	def split(self, X, y=None, groups=None):
		n_samples = len(X)
		k_fold_size=n_samples // self.n_splits
		indices = np.arange(n_samples)
		
		margin=0
		for i in range(self.n_splits):
			start=i*k_fold_size
			stop=start+k_fold_size
			mid=int(0.8 * (stop - start)) + start
			yield indices[start: mid], indices[mid + margin: stop]


class ExpandedSplit:
	def __init__(self, n_splits):
		self.n_splits = n_splits
	def get_n_splits(self, X, y, groups):
		return self.n_splits
	def split(self, X, y=None, groups=None):
		n_samples = len(X)
		k_fold_size=n_samples // self.n_splits
		indices = np.arange(n_samples)
		margin=5
		for i in range(1, self.n_splits):
			start = i*k_fold_size
			stop=start+k_fold_size
			#mid=int(0.8 * (stop - start)) + start
			#yield indices[0: mid], indices[mid + margin: stop]
			yield indices[0: start], indices[start + margin: stop]






