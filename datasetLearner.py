from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
from GraphDensity import GraphDensityCalculator
import scipy as sp
from math import log
from numpy.linalg import norm
from math import isnan
from math import isinf
import random


class datasetLearner:
	def __init__(self,train_data,train_label,test_data,test_label,hp):
		self.train_data=train_data
		self.train_label=train_label
		self.test_data=test_data
		self.test_label=test_label
		self.num_sample=train_data.shape[0]
		self.label_list=[]
		self.asked=[False for i in xrange(self.num_sample)]
		self.canTrain=False
		self.clf=LogisticRegression(C=hp.C,solver='lbfgs',warm_start=True) 
		self.gdc=GraphDensityCalculator(train_data)
		self.hp=hp
		self.lld=log(0.5)
		self.maxdis=0
		for i in xrange(self.num_sample):
			for j in xrange(i+1,self.num_sample):
				self.maxdis=max(self.maxdis,norm(self.train_data[i,:]-self.train_data[j,:]))
	def restart(self):
		self.asked=[False for i in xrange(self.num_sample)]
		self.label_list=[]
		self.canTrain=False
		self.clf=LogisticRegression(C=self.hp.C,solver='lbfgs',warm_start=True) 
		self.gdc.reset()
		self.lld=log(0.5)
	def compute_KFF(self,sid=-1):
		vals=[]
		if sid==-1:
			idxs=[i for i in xrange(self.num_sample)]
		else:
			idxs=[sid]
		if len(self.label_list)==0:

			for idx in idxs:
				vals.append(self.maxdis)
		else:
			for idx in idxs:
				dis=sys.maxint
				for l in self.label_list:
					dis=min(dis,norm(self.train_data[l,:]-self.train_data[idx,:]))
				vals.append(dis)
		if sid!=-1:
			return vals[0]
		else:
			return vals
	def compute_ENT(self,sid=-1):
		vals=[]
		if sid==-1:
			idxs=[i for i in xrange(self.num_sample)]
		else:
			idxs=[sid]
		if self.canTrain:
			vals=self.clf.predict_proba(self.train_data[idxs,:]).transpose()
			vals=sp.stats.entropy(vals)
		else:
			val=sp.stats.entropy([0.5,0.5])
			for i in xrange(len(idxs)):
				vals.append(val)
		if sid!=-1:
			return vals[0]
		else:
			return vals
	def generate_action_mat(self,idx):
		res=np.zeros((1,self.hp.actionShape))
		res[0,0:-3]=self.train_data[idx,:]
		res[0,-3]=self.compute_ENT(idx)
		res[0,-2]=self.compute_KFF(idx)
		res[0,-1]=self.gdc.getGra(idx)
		# print 'res=',res
		# raw_input('ok?')
		return res
	def add_label(self,idx):
		self.label_list.append(idx)
		self.asked[idx]=True
		self.gdc.update(idx)
		if not(self.canTrain):
			ff=False
			ft=False
			for p in self.label_list:
				if self.train_label[p]==1:
					ft=True
				if self.train_label[p]==0:
					ff=True
				if ff and ft:
					break
			if ff and ft:
				self.canTrain=True
		if self.canTrain:
			self.clf.fit(self.train_data[self.label_list,:],self.train_label[self.label_list])
		if self.canTrain:
			ld=self.clf.predict_log_proba(self.test_data)
			pred=self.clf.predict(self.test_data)
			# print ld[0:5,:]
			# print pred[0:5]
			# raw_input('ok?')
			# print ld.shape
			thislld=0
			for j in xrange(self.test_data.shape[0]):
				# print self.train_label[j]
				# if self.train_label[self.label_list[0]]==0:
				# 	thislld+=ld[j,int(self.test_label[j])]
				# else:
				
				if isnan(ld[j,int(self.test_label[j])]) or isinf(ld[j,int(self.test_label[j])]):
					# print self.train_label[self.label_list[0]],self.label_list[0]
					# print thislld,self.test_label[j],pred[j],ld[j,:]
					# raw_input('ok?')
					print 'error: infinity log prob'
					thislld-=10
				elif ld[j,int(self.test_label[j])]<=-10:
					thislld-=10
				else:
					thislld+=ld[j,int(self.test_label[j])]
			thislld/=self.num_sample
		else:
			thislld=-log(2)
		# print thislld
		self.lld=thislld
	def get_lld(self):
		return self.lld
	def generate_state(self):
		res=np.zeros((1,self.hp.stateShape))
		if self.canTrain:
			res[0,0:-4]=self.clf.coef_
			res[0,-4]=self.clf.intercept_
		res[0,-3]=self.get_lld()
		res[0,-2]=len(self.label_list)
		res[0,-1]=self.num_sample-len(self.label_list)
		return res
		# raw_input('ok?')
	def evaluate_action(self,net=None,session=None, mode='sqn'):
		if mode=='sqn':
			netinput=np.zeros((self.test_data.shape[0]-len(self.label_list),self.hp.shape))
			netinput[:,0:self.hp.stateShape]=self.generate_state()
			count=0
			# print learners[i].label_list, netinput.shape
			corres_idx=[]
			for j in xrange(self.test_data.shape[0]):
				if not(self.asked[j]):
					netinput[count,self.hp.stateShape:]=self.generate_action_mat(j)
					count+=1
					corres_idx.append(j)
			if count!=netinput.shape[0]:
				raw_input('wrong! count inconsistent')
			
			[actionVal]=session.run([net.output],feed_dict={net.x:netinput})
			
			maxact=np.argmax(actionVal)
			samplechosen=corres_idx[maxact]
			return samplechosen,actionVal[maxact],netinput[maxact,:]
		elif mode=='random':
			idx=random.randint(0,self.test_data.shape[0]-1)
			while self.asked[idx]:
				idx=random.randint(0,self.test_data.shape[0]-1)
			return idx
		elif mode=='uncertainty':
			if not(self.canTrain):
				idx=random.randint(0,self.test_data.shape[0]-1)
				while self.asked[idx]:
					idx=random.randint(0,self.test_data.shape[0]-1)
				return idx			
			else:	
			# print self.compute_ENT()
			# raw_input('ok?')
				ent=self.compute_ENT()
				idx=np.argmax(ent)
				while self.asked[idx]:
					ent[idx]=-100000
					idx=np.argmax(ent)
				
				return idx




