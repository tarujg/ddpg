""" Learn a policy using DDPG for the reach task"""
import time

from matplotlib.pyplot import plot

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gym
import pybullet
import pybulletgym.envs
from copy import deepcopy,copy

from model import Actor, Critic
from memory import ReplayBuffer,Exp
from utils import plotting

SEED = 1000
np.random.seed(SEED)

def weighSync(target, source, tau=0.001):
    """
    Function to soft update target networks
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DDPG():
    def __init__(self, env, action_dim, state_dim, device, 
                 critic_lr=3e-4, actor_lr=3e-4, gamma=0.99, batch_size=100, validate_steps = 100, max_episode_length = 150):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        param: device: The device used for training
        param: validate_steps: Number of iterations after which we evaluate trained policy 
        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.device = device
        self.eval_env = deepcopy(env)
        self.validate_steps = validate_steps
        self.max_episode_length = max_episode_length

        # actor and actor_target where both networks have the same initial weights
        self.actor = Actor(state_dim = state_dim, action_dim = action_dim).to(self.device)
        self.actor_target = deepcopy(self.actor)

        # critic and critic_target where both networks have the same initial weights
        self.critic = Critic(state_dim = state_dim, action_dim = action_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)

        # Optimizer for the actor and critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.ReplayBuffer = ReplayBuffer(buffer_size=10000, init_length=1000, state_dim=state_dim, \
                                         action_dim=action_dim, env=env, device = device)

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        weighSync(self.actor_target, self.actor)
        weighSync(self.critic_target, self.critic)

    def update_network(self, batch):
        """
        A function to update the function just once
        """

        # Sample and parse batch
        state, action, reward, state_next, done = self.ReplayBuffer.batch_sample(batch)
        
        # Predicting the next action and q_value
        action_next = self.actor_target(state_next)
        q_next = self.critic_target(state_next, action_next)
        target_q = reward + (self.gamma * done * q_next)
        
        q = self.critic(state, action)
        
        # Critic update
        self.critic.zero_grad()
        value_loss = F.mse_loss(q, target_q)
        value_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -self.critic(state,self.actor(state)).mean()
        policy_loss.backward()
        self.optimizer_actor.step()

        # Target update
        self.update_target_networks()        
        return value_loss.item(), policy_loss.item()


    def select_action(self, state, isEval):

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state).squeeze(0).detach()
        if isEval:
            return action.cpu().numpy()
        action += torch.normal(0, 0.1, size = action.shape).to(self.device)
        action = torch.clamp(action, -1., 1.).cpu().numpy()
        return action

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        value_losses, policy_losses, validation_reward, validation_steps = [],[],[],[]
        
        step, episode, episode_steps,episode_reward,state = 0, 0, 0, 0., None

        while step < num_steps:
            # reset if it is the start of episode
            if state is None:
                state = deepcopy(self.env.reset())

            action = self.select_action(state, False)
            # env response with next_state, reward, terminate_info
            state_next, reward, done, _ = self.env.step(action)
            state_next = deepcopy(state_next)

            if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                done = True

            # observe and store in replay buffer
            self.ReplayBuffer.buffer_add(Exp(state=state, action=action, reward=reward, state_next=state_next, done=done))

            # update policy based on sampled batch
            batch = self.ReplayBuffer.buffer_sample(self.batch_size)
            value_loss, policy_loss = self.update_network(batch)
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)

            # evaluate
            if step % self.validate_steps == 0:
                validate_reward, steps = self.evaluate()
                validation_reward.append(validate_reward)
                validation_steps.append(steps)
                print("[Eval {:06d}/{:06d}] Steps: {:06d}, Episode Reward:{:04f}".format(step, int(num_steps), steps, validate_reward))

            # update 
            step += 1
            episode_steps += 1
            episode_reward += reward
            state = deepcopy(state_next)

            if done: # reset at the end of episode
                #print("[Train {:06d}/{:06d}] - Episode Reward:{:04f} ".format(step, num_steps, step, episode_reward))
                episode_steps, episode_reward,state  = 0, 0., None
                episode += 1

        return value_losses, policy_losses, validation_reward, validation_steps

    def evaluate(self):
        """
        Evaluate the policy trained so far in an evaluation environment
        """        
        state,done,total_reward,steps = self.eval_env.reset(),False, 0.,0

        while not done:
            action = self.select_action(state, True)
            state_next, reward, done, _ = self.eval_env.step(action)
            total_reward+= reward
            steps+=1
            state = state_next
        return total_reward/steps, steps