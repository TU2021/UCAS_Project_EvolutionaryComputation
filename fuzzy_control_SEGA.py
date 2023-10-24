import gym
import numpy as np 

import time 
import imageio
import math
import os
from numpy import sin, cos, pi, tanh, log
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


class plant():
    def __init__(self, timeStep):
        self.env = gym.make('AcrobotContinuous-v1')
        self.state = self.env.reset()
        self.timeStep = timeStep
        self.stateTrace = []
    
    def run(self,p):
        self.env.seed(0)
        self.state = self.env.reset()
        self.stateTrace, self.imageSeq = [], []
        for k in range(self.timeStep):
            # self.env.render()   #训练时关掉渲染，快很多
            # self.imageSeq.append(self.env.render(mode='rgb_array'))
            self.action, alpha = self.fuzzy_controller(p)
            self.next_state, eDot, done, _ = self.env.step(self.action)
            self.energy_done = self.env.energy_done()
            self.state = self.next_state
            self.stateTrace.append(np.concatenate([self.state, np.array([eDot])]))
            # print(self.action)
            if self.energy_done:
                 print(self.energy_done,k)
                 # os.system("pause")
                 break
        # print(alpha)
        self.finishedTimeStep = k + 1

    def run_render(self,p):
        self.env.seed(0)
        self.state = self.env.reset()
        self.stateTrace, self.imageSeq = [], []
        self.controlSingal = []
        for k in range(self.timeStep):
            # self.env.render()   #训练时关掉渲染，快很多
            self.imageSeq.append(self.env.render(mode='rgb_array'))
            self.action, alpha = self.fuzzy_controller(p)
            self.next_state, eDot, done, _ = self.env.step(self.action)
            self.energy_done = self.env.energy_done()
            self.state = self.next_state
            self.stateTrace.append(np.concatenate([self.state, np.array([eDot])]))
            self.controlSingal.append(self.action)
            if self.energy_done:
                 print(self.energy_done,k)
                 # os.system("pause")
                 break
            self.finishedTimeStep = k + 1
        self.getGif()
        return np.array(self.stateTrace), np.array(self.controlSingal)
        
        
            
    def reward(self):
        self.stateTrace = np.array(self.stateTrace)
        # self.objFun = (abs(self.stateTrace[:,-1])).sum()
        edot = abs( self.stateTrace[:,-1] ).sum()
        # edot = np.square ( self.stateTrace[:,-1])
        # edot = abs( self.stateTrace[:,-1] )
        # self.objFun = 1 / (edot.sum()/len(edot)) + (1 - self.finishedTimeStep/500) * (1 / abs( self.stateTrace[0,-1] ))   #鼓励提前结束
        self.objFun = 1 / (edot) #鼓励提前结束
        # self.objFun = edot.sum()
        # print(self.objFun)
        return self.objFun
    
    def close(self):
        self.env.close()
        
    # def get_state(self):
    #     return self.state
    
    def getGif(self):
        imageio.mimsave('cartPoleResults.gif', self.imageSeq, 'GIF', duration = 0.02)

    def fuzzy_controller(self, p):
        m = [2,2,2,2]
        m0 = np.prod(m)
        
        s = self.state
        d0 = d1 = d2 = d3 = [None, None]

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        a10 = theta1 - math.floor((theta1+pi)/(pi*2))*pi*2
        a20 = theta1 + theta2 - math.floor((theta1+theta2+pi)/(pi*2))*pi*2
        # a20 = theta2
        # a10 = cos(theta1)
        # a20 = cos(theta1 + theta2)
        # print(a10,a20)
        av10 = dtheta1
        av20 = dtheta1 + dtheta2
        # av20 = dtheta2

        # pp  = [x * 16 - 8 for x in p[0:m0//2]]
        # coef = [x * 7.5 + 0.5 for x in p[m0//2 : m0//2+4]]

        pp  = [x  for x in p[0:m0//2]]
        coef = [x  for x in p[m0//2 : m0//2+4]]

        #  计算隶属度函数
        d0[0] = (1-tanh(coef[0]*a10))/2
        d0[1] = (1+tanh(coef[0]*a10))/2
        d1[0] = (1-tanh(coef[1]*av10))/2
        d1[1] = (1+tanh(coef[1]*av10))/2        
        d2[0] = (1-tanh(coef[2]*a20))/2
        d2[1] = (1+tanh(coef[2]*a20))/2      
        d3[0] = (1-tanh(coef[3]*av20))/2
        d3[1] = (1+tanh(coef[3]*av20))/2

        #  模糊规则
        control = 0
        sum_alpha = 0
        alpha = np.zeros(m0)
        for a in range(0, m[0]):
            for b in range(0, m[1]):
                for c in range(0, m[2]):
                    for d in range(0, m[3]):
                        j = a*m[1]*m[2]*m[3] + b*m[2]*m[3] + c*m[3] + d
                        alpha[j] = d0[a] * d1[b] * d2[c] * d3[d]
                        sum_alpha = sum_alpha+alpha
                        if j < m0/2:
                            control += alpha[j] * pp[j]
                        else:
                            control += alpha[j] * (-pp[m0-j-1])
        return control , alpha
        # return control/sum_alpha
    
