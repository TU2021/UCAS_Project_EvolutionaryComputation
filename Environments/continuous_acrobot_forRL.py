"""classic Acrobot task"""
import numpy as np
from numpy import sin, cos, pi, tanh
import math
from gym import core, spaces
from gym.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class Continuous_AcrobotEnv(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    # dt = 0.2
    dt = 0.02

    LINK_LENGTH_1 = 1.0  # [m]
    # LINK_LENGTH_2 = 1.0  # [m]
    LINK_LENGTH_2 = 2.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    # LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_COM_POS_2 = 1  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    # # RE30
    # LINK_MASS_1 = 0.37292
    # LINK_MASS_2 = 0.28852
    # LINK_LENGTH_1 = 0.07
    # LINK_LENGTH_2 = 0.12
    # LINK_COM_POS_1 = 0.06
    # LINK_COM_POS_2 = 0.04628

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    # AVAIL_TORQUE = [-1.0, 0.0, +1]

    torque_noise_max = 0.0

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.viewer = None
        high = np.array(
            [pi,pi, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        
        # ns[0] = wrap(ns[0], -pi, pi)
        # ns[1] = wrap(ns[1], -pi, pi)
        # # ns[0] = bound(ns[0], -pi, pi)
        # # ns[1] = bound(ns[1], -pi * 2 /3, pi * 2 /3)
        # ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        # ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()
        self.energy = [0,0,0]

        self.min_action = np.float32(-1.0) * 8
        self.max_action = np.float32(1.0) * 8
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), )

        
        self.sum_eDot = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,)).astype(
            np.float32
        )
        # self.state = self.np_random.uniform(low=-1, high=1, size=(4,)).astype(
        #     np.float32
        # )
        return self._get_ob()

    
    def step(self, a):
        s = self.state
        # torque = self.AVAIL_TORQUE[a]
        torque = a.clip(self.min_action, self.max_action)   # for continuous control
        # print("torque",torque)
        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        #ns[0] = wrap(ns[0], -pi, pi)
        #ns[1] = wrap(ns[1], -pi, pi)
        ns[0] = bound(ns[0], -pi, pi)
        ns[1] = bound(ns[1], -pi * 2 /3, pi * 2 /3)
        #
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        self.energy = self._energy()
        terminal = self._terminal()
        if terminal: print(f"terminal:{terminal}")
        #eDot = -1.0 if not terminal else 0.0
        
        #eDot = 1/abs(self.energy[2] - self.energy[3]) + 0.05/max(self.energy[3]*0.05,self.energy[0])
        
        #eDot = -(self.energy[2] - self.energy[3])  #
        #eDot = eDot**3/500
        
        #eDot = self.energy[2] - self.energy[3]  #反的reward，越小越好
        
        ############稠密奖励
        #print(self.energy[3] - self.energy[1])
        r1 = abs(self.energy[2] - self.energy[3])
        r2 = abs(self.energy[3] - self.energy[1])
        #print(f"{r1}  |  {r2}")
        #eDot = r1 + r2
        #eDot = math.log(r1**2) + 0.1*math.log(r2**2) #1
        #eDot = r1 + 0.1*r2 #1
        #eDot = r1**2 + 0.1*r2**2 #3
        #eDot = r1**2 + 0.5*r2**2 #4
        #eDot = math.log(r1**2) + 0.5*math.log(r2**2) #5
        
        ####
        eDot = (-r1**2 + 0.1*-r2**2)/10000 #6
        #if terminal: eDot = 0
        ####
        
        ####能量计算
        #eDot = self.energy[2] - self.energy[3]  #反的reward，越小越好
        
        
        ############角度奖励
        # theta1 = self.state[0]
        # theta2 = self.state[1]
        # v1 = self.state[2]
        # v2 = self.state[3]
        
        # threshold_1 = 0.6
        # threshold_2 = 0.5
        # if abs(theta1)<pi*threshold_1:
        #     eDot = abs(theta1)**2
        # else:
        #     eDot = abs(theta1)**2+8*(pi/4-abs(theta2))*abs((pi/4-abs(theta2)))-0.1*abs(v1)**2-0.1*abs(v2)**2
        #     if (pi/4-abs(theta2))>0:
        #         print(f"yes-{eDot}",end=" ")
        #         if eDot>0: eDot = eDot*100
        #     else:
        #         print(f"no-{eDot}",end=" ")
        # if terminal: 
        #     print(f"{theta1} | {theta2} | {v1} | {v2}")
        #     print(eDot, end="  |")
        #     eDot = abs(eDot)*(100*(abs(theta1)-pi*threshold_1)/((1-threshold_1)*pi)+1)
        #     print(eDot)
        
        #print(f"{theta1} | {theta2} | {v1} | {v2}")
        ####
        # threshold_1 = 0.7
        # threshold_2 = 0.5
        # if abs(theta1)<pi*threshold_1:
        #     eDot = theta1**2
        # elif abs(theta2)<pi*threshold_2:
        #     if ((pi-abs(theta1))/pi)*self.MAX_VEL_1 and abs(v2)<((pi-abs(theta1))/pi)*self.MAX_VEL_2:
        #         eDot = theta1**2+0.5*(abs(theta1)-pi*threshold_1)*(2*(pi-abs(theta2))**2-v1**2-v2**2)
        #     else:
        #         eDot = -abs(theta1)*max(abs(v1),abs(v2))
        # else:
        #     eDot = theta1**2+(abs(theta1)-pi*threshold_1)*(-2.5*(theta2)**2-0.2*(theta2)*v2-0.4*v1**2-0.2*v2**2)  #加了速度惩罚 1 2- 3+
        # if terminal: 
        #     print(f"{theta1} | {theta2} | {v1} | {v2}")
        #     print(eDot)
        #     eDot = abs(eDot)*(10*(abs(theta1)-pi*threshold_1)/((1-threshold_1)*pi)+1)
        ######
        #eDot = theta1**2+0.1*theta2**2-0.1*v1**2-0.1*v2**2
        
        
        
        ############稀疏奖励
        # eDot = self.energy[2] - self.energy[3]  #反的reward，越小越好
        # self.sum_eDot = self.sum_eDot + abs(eDot)
        # if terminal:
        #     eDot = 1/(self.sum_eDot)
        # else:
        #     eDot = -1
        
        return (self._get_ob(), eDot, terminal, False ,{})

    def energy_done(self):
        s = self.state
        energy = self.energy[2]
        Ek = self.energy[0]
        Ep = self.energy[1]

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        q4 = m1*lc1+m2*l1
        q5 = m2*lc2
        g = 9.8

        Etop =(q4+q5)*g

        # print(Ep,Ek)
        return bool((abs(Etop - energy) < 5 * Etop*0.01) & (Ek < 5 *Etop*0.01))
    

    def _energy(self):
        #计算时注意环境的theta和论文里的不一样，取cos时候要差一个负号
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2

        Ic1 = m1*l1*l1/3-m1*lc1*lc1
        Ic2 = m2*l2*l2/3-m2*lc2*lc2

        g = 9.8
        s = self.state
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        q1 = m1*lc1*lc1+m2*l1*l1+Ic1
        q2 = m2*lc2*lc2+Ic2
        q3 = m2*l1*lc2
        q4 = m1*lc1+m2*l1
        q5 = m2*lc2

        Ek = 0.5*(q1+q2+2*q3*cos(theta2))*dtheta1*dtheta1 + (q2+q3*cos(theta2))*dtheta1*dtheta2 + 0.5*q2*dtheta2*dtheta2
        Ep = q4*g*cos(theta1-pi) + q5*g*cos(theta1+theta2-pi)
        # print(Ek)
        Etotal = Ek + Ep
        Etop =(q4+q5)*g
        return [Ek, Ep, Etotal, Etop]

    def _get_ob(self):
        s = self.state
        # return np.array(
        #     [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        # )
        return np.array(
             [s[0],s[1], s[2], s[3]], dtype=np.float32
         )

    def _terminal(self):
        s = self.state
        energy = self.energy[2]

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        Ic1 = self.LINK_MOI
        Ic2 = self.LINK_MOI
        g = 9.8
        s = self.state

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        q1 = m1*lc1*lc1+m2*l1*l1+Ic1
        q2 = m2*lc2*lc2+Ic2
        q3 = m2*l1*lc2
        q4 = m1*lc1+m2*l1
        q5 = m2*lc2

        Etop =(q4+q5)*g
        eDot = energy-Etop
        Ep = q4*g*cos(theta1-pi) + q5*g*cos(theta1+theta2-pi)
        
        return bool((abs(eDot) < 5*Etop*0.01) & ((Etop-Ep) < 5*Etop*0.01))

    


    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2
            ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0)

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        # self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function

    Example 1 ::
        ## 2D system
        def derivs(x):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    This would then require re-adding the time variable to the signature of derivs.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]