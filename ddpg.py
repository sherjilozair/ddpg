import sys, os
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import imageio

class Memory:
    def __init__(self, env, capacity=10**6):
        self.capacity = capacity
        self.obs_sz = env.observation_space.shape[0]
        self.ac_sz = env.action_space.shape[0]
        self.obss = np.zeros((capacity, self.obs_sz))
        self.actions = np.zeros((capacity, self.ac_sz))
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
        if self.full:
            idx = np.random.randint(self.capacity-1, size=(mbsz,))
        else:
            idx = np.random.randint(self.j-1, size=(mbsz,))
        mb_obs = self.obss[idx]
        mb_nobs = self.obss[idx+1]
        mb_actions = self.actions[idx]
        mb_rewards = self.rewards[idx]
        return mb_obs, mb_nobs, mb_actions, mb_rewards


class Policy:
    def __init__(self, env):
        self.g = 0.99
        self.decay = 0.999
        self.env = env
        assert np.all(self.env.action_space.high == -self.env.action_space.low)
        self.sess = tf.Session()
        self.create_train_op()
        self.sess.run(tf.global_variables_initializer())

    def build_actor(self, s, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                h = slim.fully_connected(s, 300, scope='fc/1')
                h = slim.fully_connected(h, 300, scope='fc/2')
                h = slim.fully_connected(h, self.env.action_space.shape[0],
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                    scope='fc/3')
                return h * self.env.action_space.high

    def build_critic(self, s, a, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope('critic'):
                ho = slim.fully_connected(s, 300, scope='obs/fc/1')
                ho = slim.fully_connected(ho, 300, scope='obs/fc/2')
                ha = slim.fully_connected(a, 300, scope='act/fc/1')
                ha = slim.fully_connected(ha, 300, scope='act/fc/2')
                h = slim.fully_connected(ha + ho, 300, scope='both/fc/1')
                return slim.fully_connected(h, 1, activation_fn=None,
                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                    scope='both/fc/2')

    def create_train_op(self):
        self.s = tf.placeholder(tf.float32, (None, ) + env.observation_space.shape)
        self.ns = tf.placeholder(tf.float32, (None, ) + env.observation_space.shape)
        self.a = tf.placeholder(tf.float32, (None, ) + env.action_space.shape)
        self.r = tf.placeholder(tf.float32, (None, 1))

        self.u = self.build_actor(self.s, scope='main')
        q_main = self.build_critic(self.s, self.a, scope='main')

        u_target = self.build_actor(self.ns, scope='target')
        q_target = self.build_critic(self.ns, u_target, scope='target') * self.g + self.r

        self.q_loss = tf.reduce_mean(0.5 * (q_main - q_target)**2)
        self.a_loss = tf.reduce_mean(- self.build_critic(self.s, self.u, scope='main'))

        self.train_actor = tf.train.AdamOptimizer(1e-4).minimize(self.a_loss,
            var_list=slim.get_variables(scope="main/actor"))
        self.train_critic = tf.train.AdamOptimizer(1e-3).minimize(self.q_loss,
            var_list=slim.get_variables(scope="main/critic"))

        updates = []
        for f in ['actor', 'critic']:
            for s, t in zip(
                slim.get_variables(scope="main/{}".format(f)),
                slim.get_variables(scope="target/{}".format(f))):
                new_t = t * self.decay + s * (1 - self.decay)
                updates.append(tf.assign(t, new_t))
        ops = [self.train_actor, self.train_critic] + updates
        self.train_op = tf.group(*ops)

    def act(self, s):
        return self.sess.run(self.u, {self.s: s[None, :]})[0]

    def update(self, s, ns, a, r):
        _, q_loss, a_loss = self.sess.run(
            [self.train_op, self.q_loss, self.a_loss],
            {self.s: s, self.ns: ns, self.a: a, self.r: r})
        return q_loss, a_loss


class Agent:
    def __init__(self, expname, policy, memory, eplen=100, iters=50000, mbsz=64, save_every=50):
        self.expname = expname
        self.policy = policy
        self.memory = memory
        self.eplen = eplen
        self.iters = iters
        self.mbsz = mbsz
        self.save_every = save_every

    def loop(self):
        for e in xrange(self.iters):
            obs = env.reset()
            done = False
            rtrn = 0.
            q_loss = 0.
            a_loss = 0.
            frames = []
            for j in xrange(self.eplen):
                action = policy.act(obs)
                next_obs, reward, done, info = env.step(action)
                if e % self.save_every == 0:
                    frames.append(env.render(mode='rgb_array'))
                rtrn += reward
                self.memory.add(obs, action, reward)
                obs = next_obs

                if self.memory.j > self.mbsz:
                    batch = self.memory.get_batch(self.mbsz)
                    ql, al = policy.update(*batch)
                    q_loss += ql
                    a_loss += al

            if e % self.save_every == 0:
                imageio.mimwrite("{}/gifs/{}.gif".format(self.expname, e), frames, fps=36)

            print "Episode {}, Return: {}, (Q, A)-loss: {}, {}".format(e, rtrn,
                q_loss/self.eplen, a_loss/self.eplen)


if __name__ == '__main__':
    expname = strftime("%Y.%m.%d|%H.%M.%S", gmtime())
    if not os.path.exists(expname):
        os.makedirs('{}/gifs'.format(expname))

    envname = sys.argv[1]
    eplen = int(sys.argv[2])

    env = gym.make(envname)
    memory = Memory(env, capacity=10**6)
    policy = Policy(env)
    agent = Agent(expname, policy, memory, eplen=eplen)
    agent.loop()
