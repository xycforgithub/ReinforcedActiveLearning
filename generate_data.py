import numpy as np
import cPickle as pickle
from math import sqrt
# import matplotlib.pyplot as plt
#create training data

num_data=100
num_dataset=90
data_dim=3	
train_datas=[]
train_labels=[]
test_datas=[]
test_labels=[]
for ds in xrange(num_dataset):
	mean1=np.random.randn(data_dim)/2+0.7
	mean2=np.random.randn(data_dim)/2-0.7
	cov1=np.random.randn(data_dim,data_dim)/2
	cov1=cov1.dot(cov1.transpose())
	cov2=np.random.randn(data_dim,data_dim)/2
	cov2=cov2.dot(cov2.transpose())
	# print mean1
	data1=np.random.multivariate_normal(mean1,cov1,num_data/2)
	data2=np.random.multivariate_normal(mean2,cov2,num_data/2)
	label=np.ones(num_data)
	label[num_data/2:num_data]=0
	# print data1.shape
	data=np.vstack([data1,data2])
	randperm=np.random.permutation(num_data-1)
	randperm=np.hstack((0,randperm))
	# print data.shape
	data=data[randperm,:]
	label=label[randperm]
	train_datas.append(data)
	train_labels.append(label)



	data1=np.random.multivariate_normal(mean1,cov1,num_data/2)
	data2=np.random.multivariate_normal(mean2,cov2,num_data/2)
	label=np.ones(num_data)
	label[num_data/2:num_data]=0
	# print data1.shape
	data=np.vstack([data1,data2])
	randperm=np.random.permutation(num_data)
	# print data.shape
	data=data[randperm,:]
	label=label[randperm]
	test_datas.append(data)
	test_labels.append(label)
dataset={'train_datas':train_datas,'train_labels':train_labels,'test_datas':test_datas,'test_labels':test_labels}
pickle.dump(dataset,open('dataset.dat','w'))
