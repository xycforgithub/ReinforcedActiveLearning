import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import random
from math import log
from datasetLearner import datasetLearner
import numpy as np

class SampleQNetwork:
    def __init__(self,numInput,hp,num_hidden=50,learning_rate=0.0001):
        self.x=tf.placeholder(tf.float32,shape=[None,numInput],name='Input')
        self.W1=self.weight_variable([numInput,num_hidden],name='weight1')
        self.b1=self.bias_variable([num_hidden],name='bias1')
        self.h=tf.nn.softplus(tf.matmul(self.x,self.W1)+self.b1, name='hidden')
        self.W2=self.weight_variable([num_hidden,1],name='weight2')
        self.b2=self.bias_variable([1],name='bias2')
        self.output=tf.matmul(self.h,self.W2)+self.b2

        # Funciton target, can be Q-learning, Sarsa, etc.
        self.targetQ=tf.placeholder(shape=[None,1],dtype=tf.float32,name='targetQ')
        self.error=tf.square(self.targetQ-self.output)

        self.loss=tf.reduce_mean(self.error)

        self.train_step=tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.hp=hp
        
    def weight_variable(self,shape,name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial,name=name)
    
    def bias_variable(self,shape,name):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial,name=name)
    def test_network(self,train_data,train_label,test_data,test_label,session):
        num_sample=train_data.shape[0]
        learner=datasetLearner(train_data,train_label,test_data,test_label,self.hp)
        llds=[]
        action=random.randint(0,num_sample-1)
        for iteration in xrange(self.hp.episodeLength):
            learner.add_label(action)
            llds.append(learner.get_lld())
            action,_,_=learner.evaluate_action(self,session)



        return llds
    def save_model(self,path,sess):
        '''
        Stores the network weights in different numpy arrays
        '''
        W1=sess.run(self.W1)
        W2=sess.run(self.W2)
        b1=sess.run(self.b1)
        b2=sess.run(self.b2)
        
        with open(filename,'wb') as f:
            np.savez(f,W1=W1,W2=W2,b1=b1,b2=b2)
            


    