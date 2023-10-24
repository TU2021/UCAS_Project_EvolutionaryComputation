import geatpy as ea
from fuzzy_control_SEGA import plant
import time 
import numpy as np
import matplotlib.pyplot as plt

class OptimizeFuzzyControl(ea.Problem):                
    def __init__(self):
        # self.p = [-3.8466, -4.3624, 0.6857, 6.0230, -4.2999, 5.8667, -4.2999, 5.4916, 2.6111, 4.5621, 9.7611, 6.7104] #待优化的参数表
        name = 'OptimizeFuzzyControl'                  
        M = 1
        maxormins = [-1]    #1表示最小化，-1表示最大化
        # maxormins = [1]    #1表示最小化，-1表示最大化
        Dim = 12
        varTypes = [0] * Dim
        lb = [-8] * 8 + [0.5] * 4  # k的值有限制
        # # lb = [-10] * Dim
        ub = [8] * 8 + [8] * 4
        # lb = [0] * Dim
        # ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):                 
        Vars = pop.Phen 
        objvalues = np.zeros((NIND,1))
        for i in range(NIND):
            # self.p = Vars[i]
            # print(len(self.p))
            cartPole.run(Vars[i])
            objvalues[i] = cartPole.reward()
            # print(Vars[i])
        pop.ObjV = objvalues

if __name__ == '__main__':
    cartPole = plant(500)
    problem = OptimizeFuzzyControl()                                
    Encoding = 'RI'    
    # Encoding = 'BG' # 'BG'表示采用二进制/格雷编码
    NIND = 100   # 种群数
    

    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)    
    population = ea.Population(Encoding, Field, NIND)

    # myAlgorithm = ea.soea_SEGA_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem,population)
    # myAlgorithm = ea.soea_SGA_templet(problem, population)  

    myAlgorithm.MAXGEN = 100

    myAlgorithm.recOper.XOVR = 0.9 #交叉概率 SEGA需要较大的交叉和变异概率
    myAlgorithm.mutOper.Pm = 0.1 #变异概率
    # if Encoding == 'RI':
    #     myAlgorithm.recOper = ea.Recndx(XOVR = 1) # 生成正态分布交叉算子对象

    myAlgorithm.drawing = 1
    myAlgorithm.logTras = 1                                      

    # [population, obj_trace, var_trace] = myAlgorithm.run() 
    [NDSet, population] = myAlgorithm.run() #
    NDSet.save() # 把非支配种群的信息保存到文件中

    cartPole.close()
    print(NDSet.Phen)

    best_p = NDSet.Phen.flatten()
    cartPole_best = plant(500)

    state_tra, control_singal = cartPole_best.run_render(best_p)
    cartPole_best.close()

    #展示结果 
    time_list = [t * 0.02 for t in range(len(state_tra))]   # delta t = 0.02
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(time_list, state_tra[:,0], label=r'$\theta_1$')
    axes[0, 0].plot(time_list, state_tra[:,1], label=r'$\theta_2$')
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r'$\theta$'+"(rad)")
    axes[0, 0].legend()

    axes[0, 1].plot(time_list, state_tra[:,2], label=r'$\omega_1$')
    axes[0, 1].plot(time_list, state_tra[:,3], label=r'$\omega_2$')
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r'$\omega$'+"(rad/s)")
    axes[0, 1].legend()

    axes[1, 0].plot(time_list, state_tra[:,4],color='black')
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r'$\Delta$'+"Energy(J)")

    axes[1, 1].plot(time_list, control_singal,color='black')
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("Torque(Nm)")

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

    

    # """===========================输出结果========================"""
    # print('用时：%s 秒' % myAlgorithm.passTime)
    # print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')

    # # if myAlgorithm.log is not None and NDSet.sizes != 0:
    # #     print('GD', myAlgorithm.log['gd'][-1])
    # #     print('IGD', myAlgorithm.log['igd'][-1])
    # #     print('HV', myAlgorithm.log['hv'][-1])
    # #     print('Spacing', myAlgorithm.log['spacing'][-1])
    # """======================进化过程指标追踪分析=================="""
    # # metricName = [['igd'], ['hv']]
    # # Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in
    # # range(len(metricName))]).T
    # # # 绘制指标追踪分析图
    # # ea.trcplot(Metrics, labels=metricName, titles=metricName)


    # cartPole = plant(200)
    # pidController = cartPole.fuzzy_controller()
    # cartPole.run(pidController)
    # print("objective values: ", cartPole.reward())
    # cartPole.getGif()
    # cartPole.close()