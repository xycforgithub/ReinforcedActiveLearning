import numpy as np
from math import log
from numpy.linalg import norm
import sys
import scipy as sp

class HyperParameter:
	num_hidden=50
	learning_rate=0.0001
	batchSize=32
	expSize=32
	stateShape=7
	actionShape=6
	
	max_iter=500000
	startingEpsilon=1000
	endEpsilon=0.1
	maxiterationEpsilon=100000
	C=1
	episodeLength=40
	gamma=0.95
	testeps=0.05
	testInterval=10000
	num_test=20

	def __init__(self):
		self.shape=self.actionShape+self.stateShape
		pass


class experience:
	def __init__(self, hp):
		self.exp=[]
		self.batchSize=hp.batchSize
		self.expSize=hp.expSize
		self.currentIndex=0
		self.shape=hp.shape
		self.stateShape=hp.stateShape
		self.actionShape=hp.actionShape
	def add(self,item):
		if len(self.exp)<self.expSize:
			self.exp.append(item)
		else:
			self.exp[self.currentIndex]=item
			self.currentIndex=(self.currentIndex+1) % self.expSize
	def can_produce_batch(self):
		if len(self.exp)>=self.batchSize:
			return True
		else:
			return False
	def produce_batch(self):
		batch=np.zeros((self.batchSize,self.shape))
		inds=np.random.randint(0,len(self.exp))
		return batch[-1]
def epsilonCalc(iteration,hp):
        #Very simple linear decay of the epsilon
    return hp.startingEpsilon-iteration*((hp.startingEpsilon-hp.endEpsilon)/hp.maxiterationEpsilon) if iteration < int(hp.maxiterationEpsilon) else 0.1
def calculateEnt(clf,data):
	p=clf.predict_proba(data)
	return -p[0]*log(p[0])-p[1]*log(p[1])


