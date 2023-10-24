#import gymnasium as gym
import gym
from stable_baselines3 import SAC,PPO,DQN,common
import numpy as np
import imageio
from stable_baselines3.common.logger import configure
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter   


class rl:
    def __init__(self):
        #self.env = gym.make("Acrobot-v1")
        self.env = gym.make("AcrobotContinuous-v1")
        self.save_path={'PPO':'ppo_pendulum','SAC':'sac_pendulum','DQN':'dqn_pendulum'}
        self.implement={'PPO':PPO,'SAC':SAC,'DQN':DQN}
        
        
    def train(self,algorithm):
        print("start training!")
        tmp_path = f"./log/{self.save_path[algorithm]}/"
        writer = SummaryWriter(f'./log/{self.save_path[algorithm]}/') #reward
        #n_step = 500
        model = self.implement[algorithm]("MlpPolicy", self.env, verbose=1, learning_rate=0.0003,gamma=1)
        # set up logger
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)     
        model.set_random_seed(seed=0)
        for i in tqdm(range(2000)):
            model.learn(total_timesteps=1000, log_interval=4)
            if i%20==0:
                model.save(self.save_path[algorithm])
            if i%5==0:
                return_ = 0
                eval_len = 30
                terminate_ = 0
                for k in range(eval_len):
                    total_reward,terminate = self.eval(algorithm,if_render=0,model=model)
                    return_ = return_ + total_reward
                    terminate_ = terminate_+terminate
                self.eval(algorithm,if_render=1,model=model)
                writer.add_scalar("return", return_/eval_len, (i+1)*1000)
                writer.add_scalar("success rate", terminate_/eval_len, (i+1)*1000)
                print(f"\r\n success rate:{terminate_}")
        #average_reward=common.evaluation.evaluate_policy(model,self.env,n_eval_episodes=10)
        del model # remove to demonstrate saving and loading
    
    def eval(self,algorithm,if_render=0,model=None):
        total_reward = 0
        self.controlSingal,self.stateTrace, self.imageSeq = [], [], []
        if model == None: model = self.implement[algorithm].load(self.save_path[algorithm]) 
        obs = self.env.reset()
        #print("start eval!")
        for k in range(500):
            if if_render: self.imageSeq.append(self.env.render(mode='rgb_array'))  #self.env.render(mode='rgb_array')
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward = total_reward+reward
            #print(reward)
            if if_render:
                self.stateTrace.append(np.concatenate([obs, np.array([reward])]))
                self.controlSingal.append(action)
            if terminated: break
        #print("start render!")
        if if_render: self.getGif()
        return total_reward,terminated
    
    def eval_render(self,algorithm,if_render=0,model=None):
        total_reward = 0
        self.controlSingal,self.stateTrace, self.imageSeq = [], [], []
        if model == None: model = self.implement[algorithm].load(self.save_path[algorithm]) 
        obs = self.env.reset()
        #print("start eval!")
        for k in range(1000):
            if if_render: self.imageSeq.append(self.env.render(mode='rgb_array'))  #self.env.render(mode='rgb_array')
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward = total_reward+reward
            #print(reward)
            if if_render:
                self.stateTrace.append(np.concatenate([obs, np.array([reward])]))
                self.controlSingal.append(action)
            if terminated: break
        #print("start render!")
        if if_render: self.getGif()
        return np.array(self.stateTrace), np.array(self.controlSingal),total_reward
    
    def getGif(self):
        imageio.mimsave('acrobot new.gif', self.imageSeq, 'GIF', duration = 0.02)
        
def plot(rl_,algorithm):
    state_tra, control_singal, _ = rl_.eval_render(algorithm,1)
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
    plt.savefig("./rl.png")
    # 显示图形
    plt.show()
        
if __name__=="__main__":
    algorithm = "SAC"
    rl_ = rl()
    #rl_.train(algorithm)
    #rl_.eval(algorithm,1)

    plot(rl_,algorithm)