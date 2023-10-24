from fuzzy_control_SEGA import plant
import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, num_vars):
        mean1 = 0  # 第一个均值
        std_dev1 = 3  # 第一个方差
        mean2 = 4  # 第二个均值
        std_dev2 = 3 # 第二个方差
        self.position = np.zeros(num_vars)  # 创建零向量用于存储初始化后的数值
        self.position[:8] = np.random.normal(mean1, std_dev1, 8)
        self.position[num_vars-4:] = np.random.normal(mean2, std_dev2, 4)
        self.velocity = np.random.normal(0.4, 0.2, num_vars)
        self.personal_best_position = None
        self.personal_best_fitness = np.inf
        self.inertia_weight = None  # 添加粒子的惯性权重变量
        self.cognitive_weight = None  # 添加粒子的认知权重变量
        self.social_weight = None  # 添加粒子的社会权重变量

class PSO:
    def __init__(self, problem, num_particles, max_iterations):
        self.problem = problem
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.max_velocity = [1.2] * 8 + [0.6] * 4
        self.min_velocity = [-1.2] * 8 + [-0.6] * 4

    def optimize(self):
        num_vars = len(self.problem.lb)
        particles = [Particle(num_vars) for _ in range(self.num_particles)]
        global_best_position = None
        global_best_fitness = np.inf

        # 为可视化设置
        fitness_history = []
        fitness_average_history = []
        fitness_variance_history = []
        for iteration in range(self.max_iterations):
            fitness_average = []
            for particle in particles:
                fitness = self.problem.evaluate(particle.position)
                fitness_average.append(-fitness)
                if fitness < particle.personal_best_fitness:
                    particle.personal_best_position = particle.position
                    particle.personal_best_fitness = fitness

                if fitness < global_best_fitness:
                    global_best_position = particle.position
                    global_best_fitness = fitness

                # 计算自适应权重
                if particle.personal_best_fitness != np.inf:
                    particle.inertia_weight = 1 / (np.abs(particle.personal_best_fitness)*20000)
                if global_best_fitness != np.inf:
                    particle.cognitive_weight = 1 / (np.abs(global_best_fitness)*15000)
                particle.social_weight = 1 / (np.abs(fitness) * 15000)

                # 更新粒子速度
                particle.velocity = (particle.inertia_weight * particle.velocity +
                                     particle.cognitive_weight * np.random.rand(num_vars) *
                                     (particle.personal_best_position - particle.position) +
                                     particle.social_weight * np.random.rand(num_vars) *
                                     (global_best_position - particle.position))
                # print(np.random.rand(num_vars))
                # input()

                # 限制速度在最大值和最小值之间
                particle.velocity = np.maximum(particle.velocity, self.min_velocity)
                particle.velocity = np.minimum(particle.velocity, self.max_velocity)

                # 更新粒子位置
                particle.position += particle.velocity
                particle.position = np.maximum(particle.position, self.problem.lb)
                particle.position = np.minimum(particle.position, self.problem.ub)

            # 更新可视化数据
            temp = sum(fitness_average) / len(fitness_average)
            print(iteration, -global_best_fitness, temp)
            print(particle.position)
            print(particle.velocity)
            print()
            fitness_history.append(-global_best_fitness)
            fitness_average_history.append(temp)
            fitness_variance_history.append(np.var(fitness_average))

        # 绘制粒子最佳适应度的变化图
        x = range(len(fitness_history))
        plt.plot(x, fitness_history, label='Best Objective Value', color = "orange")
        plt.plot(x, fitness_average_history, label='Average Objective Value', color = "blue")
        plt.legend()
        plt.fill_between(x, fitness_average_history - np.sqrt(fitness_variance_history), fitness_average_history + np.sqrt(fitness_variance_history), color='blue', alpha=0.2)
        plt.xlabel("Iteration Number")
        plt.ylabel("Fitness Value")
        plt.title("Trace Plot")   
        plt.show()
        return global_best_position

class OptimizeFuzzyControl():
    def __init__(self):
        self.name = 'OptimizeFuzzyControl'
        self.M = 1
        self.maxormins = [1]
        self.Dim = 12
        self.varTypes = [0] * self.Dim
        self.lb = [-8] * 8 + [0.5] * 4
        self.ub = [8] * 8 + [8] * 4
        self.lbin = [1] * self.Dim
        self.ubin = [1] * self.Dim

    def evaluate(self, position):
        cartPole = plant(500)
        cartPole.run(position)
        fitness = -cartPole.reward()
        cartPole.close()
        return fitness

if __name__ == '__main__':
    np.random.seed(667)
    problem = OptimizeFuzzyControl()
    num_particles = 30
    max_iterations = 200

    pso = PSO(problem, num_particles, max_iterations)
    best_position = pso.optimize()

    cartPole_best = plant(500)
    state_tra, control_signal = cartPole_best.run_render(best_position)
    cartPole_best.close()
