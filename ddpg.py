import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
from collections import namedtuple

class Memory:
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.obs_sz = env.observation_space.shape[0]
        self.ac_sz = env.action_space.shape[0]
        self.obss = np.zeros((capacity, self.obs_sz))
        self.actions = np.zeros((capacity, self.ac_sz), dtype=np.uint8)
        self.rewards = np.zeros((capacity, 1))
        self.j = 0
        self.full = False

    def add(self, obs, action, reward):
        self.obss[self.j] = obs
        self.actions[self.j] = action
        self.rewards[self.j] = reward
        self.j = (self.j + 1) % self.capacity
        if self.j == 0:
            self.full = True

    def get_batch(self, mbsz=64):
        idx = np.random.randint(self.capacity-1, size=(mbsz,))
        mb_obs = self.obss[idx]
        mb_nobs = self.obss[idx+1]
        mb_actions = self.actions[idx]
        mb_rewards = self.rewards[idx]
        return mb_obs, mb_nobs, mb_actions, mb_rewards


class Policy:
    def __init__(self, env):
        self.g = 0.99
        self.env = env
        self.sess = tf.Session()
        self.create_train_op()
        self.sess.run(tf.global_variables_initializer())

    def build_actor(self, s, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                h = slim.fully_connected(s, 64, scope='fc/1')
                h = slim.fully_connected(h, 64, scope='fc/2')
                u = 2 * slim.fully_connected(h,
                    self.env.action_space.shape[0],
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                    scope='fc/3')
                return u

    def build_critic(self, s, a, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('critic'):
                ho = slim.fully_connected(s, 64, scope='obs/fc/1')
                ho = slim.fully_connected(ho, 64, scope='obs/fc/2')
                ha = slim.fully_connected(a, 64, scope='act/fc/1')
                ha = slim.fully_connected(ha, 64, scope='act/fc/2')
                h = slim.fully_connected(ha + ho, 64, scope='both/fc/1')
                c = slim.fully_connected(h, 1, activation_fn=None,
                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                    scope='both/fc/2')
                return c


    def create_train_op(self):
        self.s = tf.placeholder(tf.float32, (None, ) + env.observation_space.shape)
        self.ns = tf.placeholder(tf.float32, (None, ) + env.observation_space.shape)
        self.a = tf.placeholder(tf.float32, (None, ) + env.action_space.shape)
        self.r = tf.placeholder(tf.float32, (None, 1))

        u_target = self.build_actor(self.s, scope='target')
        q_target = self.r + self.g * self.build_critic(self.s, u_target, scope='target')
        q_main = self.build_critic(self.s, self.a, scope='main')
        self.q_loss = tf.reduce_mean((q_main - q_target)**2)

        self.u = self.build_actor(self.s, scope='main')
        self.a_loss = tf.reduce_mean(- self.build_critic(self.s, self.u, scope='main'))

        self.train_critic = tf.train.AdamOptimizer(1e-3).minimize(self.q_loss,
            var_list=slim.get_variables(scope="main/critic"))
        self.train_actor = tf.train.AdamOptimizer(1e-4).minimize(self.a_loss,
            var_list=slim.get_variables(scope="main/actor"))
        self.train_op = tf.group(*[self.train_actor, self.train_critic])


    def act(self, s):
        return self.sess.run(self.u, {self.s: s[None, :]})[0]

    def update(self, s, ns, a, r):
        _, q_loss, a_loss = self.sess.run(
            [self.train_op, self.q_loss, self.a_loss],
            {self.s: s, self.ns: ns, self.a: a, self.r: r})
        return q_loss, a_loss

class Agent:
    def __init__(self, policy, memory, iters=10000):
        self.policy = policy
        self.iters = iters

    def loop(self, ep_len=100):
        for e in xrange(self.iters):
            obs = env.reset()
            done = False
            rtrn = 0.
            loss = 0.
            for j in xrange(ep_len):
                action = policy.act(obs)
                next_obs, reward, done, info = env.step(action)
                rtrn += reward
                memory.add(obs, action, reward)
                obs = next_obs
                if memory.full:
                    batch = memory.get_batch()
                    ql, al = policy.update(*batch)
                    loss += ql + al

            print "Episode {}, Return: {}, Loss: {}".format(e, rtrn/ep_len, loss/ep_len if memory.full else "N/A")

if __name__ == '__main__':
    env = gym.make('Reacher-v1')
    capacity = 10**4
    memory = Memory(capacity, env)
    policy = Policy(env)
    agent = Agent(policy, memory)
    agent.loop()
