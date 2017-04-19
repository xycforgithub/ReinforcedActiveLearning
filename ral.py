# import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import cPickle as pickle
from sqn import SampleQNetwork
from sklearn.linear_model import LogisticRegression
import random
from GraphDensity import GraphDensityCalculator
from math import log
from datasetLearner import datasetLearner
import os

# Load datasets
datasets=pickle.load(open('dataset.dat'))
train_datas=datasets['train_datas']
test_datas=datasets['test_datas']
train_labels=datasets['train_labels']
test_labels=datasets['test_labels']
final_train_data=train_datas[-1]
final_train_label=train_labels[-1]
final_test_data=test_datas[-1]
final_test_label=test_labels[-1]
train_datas=train_datas[0:-1]
train_labels=train_labels[0:-1]
test_datas=test_datas[0:-1]
test_labels=test_labels[0:-1]
numDataset=len(train_datas)

exprNum=1
test_name='expr%d' % (exprNum)
os.system('mkdir %s'% (test_name))
util=__import__('util%d' % (exprNum))


# initialize hyper parameter and network
hp=util.HyperParameter()
session=tf.InteractiveSession()
net= SampleQNetwork(numInput=hp.shape, hp=hp,num_hidden=hp.num_hidden,learning_rate=hp.learning_rate)
session.run(tf.global_variables_initializer())


# final_learner=datasetLearner(final_train_data,final_train_label,final_test_data,final_test_label,hp)
# action=random.randint(0,final_test_data.shape[0]-1)
# random_llds=np.zeros((hp.num_test,hp.episodeLength))
# for test in xrange(hp.num_test):
# 	for i in xrange(hp.episodeLength):
# 		final_learner.add_label(action)
# 		random_llds[test,i]=final_learner.get_lld()
# 		action=final_learner.evaluate_action(mode='random')
# 	final_learner.restart()
# print 'random score:',np.mean(random_llds,axis=0)
# pickle.dump(np.mean(random_llds,axis=0),open('rand_score.dat','w'))
# final_learner.restart()
# ent_llds=np.zeros((hp.num_test,hp.episodeLength))
# for test in xrange(hp.num_test):
# 	action=random.randint(0,final_test_data.shape[0]-1)
# 	# print action,',',
# 	for i in xrange(hp.episodeLength):
# 		final_learner.add_label(action)
# 		ent_llds[test,i]=final_learner.get_lld()
# 		action=final_learner.evaluate_action(mode='uncertainty')
# 		# print action,',',
# 	# print final_train_label[final_learner.label_list]
# 	final_learner.restart()
# print 'uncertainty score:',np.mean(ent_llds,axis=0)
# pickle.dump(np.mean(ent_llds,axis=0),open('unc_score.dat','w'))
# # print ent_llds
# raw_input('ok?')



#init learners
learners=[]
for i in xrange(numDataset):
	learners.append(datasetLearner(train_datas[i],train_labels[i],test_datas[i],test_labels[i],hp))
# print test_labels[0]
# raw_input('ok')

# print numDataset,hp.actionShape
actionMat=np.zeros((numDataset,hp.actionShape))
thisAction=[]
lastlld=np.ones(numDataset)*log(0.5)

#Random select action
for i in xrange(numDataset):
	idx=random.randint(0,train_datas[i].shape[0]-1)
	
	thisAction.append(idx)
	actionMat[i,:]=learners[i].generate_action_mat(idx)
# Get initial state
statemat=np.zeros((numDataset, hp.stateShape))
for i in xrange(numDataset):
	statemat[i,:]=learners[i].generate_state()	


for iteration in xrange(hp.max_iter):



	# print 'statemat=',statemat
	# raw_input('ok?')
	# Take actions in thisAction
	thislld=np.zeros(numDataset)
	for i in xrange(numDataset):
		learners[i].add_label(thisAction[i])
		# print 'learner ',i, ' lld=',learners[i].get_lld()
		thislld[i]=learners[i].get_lld()
	if iteration % hp.episodeLength==0 and iteration>0:
		# Restart
		for i in xrange(numDataset):
			learners[i].restart()
	# Generate state
	oldstate=statemat
	statemat=np.zeros((numDataset, hp.stateShape))
	for i in xrange(numDataset):
		statemat[i,:]=learners[i].generate_state()

	# Clear thisAction and fill with next step actions
	thisAction=[]
	reward=thislld-lastlld
	reward=reward.reshape((-1,1))
	# with open('./%s/test_lld.txt' % (test_name),'a') as f:
	# 	f.write('iteration %d, mean error = %f\n' % (iteration, np.mean(thislld)))
	# 	print 'mean loss:',np.mean(thislld)


	# print 'statemat=',statemat
	# raw_input('ok?')

	# Choose next action using epsilon greeedy
	eps=util.epsilonCalc(iteration,hp)
	nextAction=np.zeros((numDataset,hp.actionShape))
	maxScore=np.zeros((numDataset,1))
	for i in xrange(numDataset):
		if np.random.rand()<eps:
			# print 'random action'
			idx=random.randint(0,learners[i].num_sample-1)
			while learners[i].asked[idx]:
				idx=random.randint(0,learners[i].num_sample-1)
			nextAction[i,:]=learners[i].generate_action_mat(idx)
			# print nextAction[i,:]
			# raw_input('ok?')
			thisAction.append(idx)
			netinput=np.zeros((1,hp.shape))
			netinput[0,0:hp.stateShape]=statemat[i,:]
			netinput[0,hp.stateShape:]=nextAction[i,:]
			[maxScore[i,0]]=session.run([net.output],feed_dict={net.x:netinput})
			# print 'here'
			
		else:
			# print 'greedy action'
			samplechosen, targetScore,maxinput=learners[i].evaluate_action(net,session)
			maxScore[i,0]=targetScore
			thisAction.append(samplechosen)
			# print maxinput.shape
			nextAction[i,:]=maxinput[hp.stateShape:]
	if iteration % hp.episodeLength==0 and iteration>0: # Episode terminates
		target=reward
	else:
		target=reward+hp.gamma*maxScore

	# Train step
	netinput=np.hstack((oldstate,actionMat))
	# print 'target=',target
	# print 'netinput[0,:]=',netinput[0,:]
	# print train_datas[0]
	_,loss,output=session.run([net.train_step,net.loss,net.output],feed_dict={net.x:netinput,net.targetQ:target})
	# print np.array(output).reshape([1,-1])
	with open('./%s/test_lld.txt' % (test_name),'a') as f:
		f.write('iteration %d, mean error = %f, loss = %f\n' % (iteration, np.mean(thislld),loss))
	print 'iteration ',iteration,' loss=',loss,'reward=',np.mean(reward), 'mean error=',np.mean(thislld)

	# Do testing
	if iteration % hp.testInterval ==0 and iteration>0:
		testresult=np.zeros((hp.num_test,hp.episodeLength))
		for testiter in xrange(hp.num_test):
			testresult[testiter,:]=net.test_network(final_train_data,final_train_label,final_test_data,final_test_label,session)
		with open('./%s/test_results.txt' % (test_name),'a') as f:
			print >>f, 'Test at iteration %d: mean error = ' % (iteration), np.mean(testresult,axis=0)
			print 'Test loss:',np.mean(testresult,axis=0)
	# raw_input('ok?')
	
	actionMat=nextAction
	if iteration % hp.episodeLength==0 and iteration>0: # Episode terminates
		lastlld=-log(2)
	else:
		lastlld=thislld
	





