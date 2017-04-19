import numpy as np
from math import log
from numpy.linalg import norm
import sys
from math import exp


class GraphDensityCalculator:
	def __init__(self,data,k=10,sigma=1):
		n=data.shape[0]
		dis=np.zeros((n,n))
		for i in xrange(n):
			for j in xrange(n):
				dis[i,j]=norm(data[i,:]-data[j,:])
		P=np.zeros((n,n))
		W=np.zeros((n,n))
		for i in xrange(n):
			orders=np.argsort(dis[i,:])
			for j in xrange(k):
				P[i,j]=1
		for i in xrange(n):
			for j in xrange(n):
				P[i,j]=max(P[i,j],P[j,i])
				W[i,j]=P[i,j]*exp(-dis[i,j]/(2*sigma*sigma))
		Gra=np.sum(W,axis=1)/np.sum(P,axis=1)
		self.initGra=np.sum(W,axis=1)/np.sum(P,axis=1)
		self.P=P
		self.W=W
		self.Gra=Gra
		self.n=n

	def update(self,sid):
		for j in xrange(self.n):
			if j!=sid:
				self.Gra[j]=self.Gra[j]-self.Gra[sid]*self.P[sid,j]
		self.Gra[sid]=0
	def getGra(self,sid):
		if sid==-1: 
			return self.Gra
		else:
			return self.Gra[sid]
	def reset(self):
		for i in xrange(self.n):
			self.Gra[i]=self.initGra[i]

