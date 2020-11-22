from collections import deque, namedtuple
import random
import torch

Exp = namedtuple('Exp', 'state, action, reward, state_next, done')

class ReplayBuffer():
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env, device):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        param: device : device 
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.buffer = deque(maxlen=self.buffer_size)
        self.device = device

        if self.env:
            self.buffer_populate()
    
    def buffer_add(self, exp):
        """
        A function to add a tuple to the buffer
        param: exp : A namedtuple consisting of state, action, reward , next state and done flag
        """
        self.buffer.append(exp)

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        raise random.sample(self.buffer, N)

    def buffer_populate(self):
        """
        Populates buffer based on random actions taken initially with init_length entries
        """
        while True:

            done = False
            state = self.env.reset()

            while not done:
                
                # Sample random action
                action = self.env.action_space.sample()
                state_next, reward, done, _ = self.env.step(action)

                # Add random action and entries to buffer
                self.buffer_add(Exp(state=state, action=action, reward=reward, state_next=state_next, done=done))
                
                state = state_next
                if len(self.buffer) >= self.init_length:
                    return


    def batch_sample(self, batch):
        """
        A function to create a batch of tensors for minibatch samples
        param: batch: a minibatch list consisting of list consisting of state, action, reward , next state and done flag
        """
        batch_size = len(batch)
        state, action, reward, state_next, done = [],[],[],[],[]
        for element in batch:
            state.append(torch.FloatTensor(getattr(element,state)))
            action.append(torch.FloatTensor(getattr(element,action)))
            reward.append(torch.FloatTensor(getattr(element,reward)))
            state_next.append(torch.FloatTensor(getattr(element,state_next)))
            done.append(torch.FloatTensor(0. if getattr(element,done) else 1.))

        # Prepare and validate parameters.
        state = torch.cat(state).reshape(batch_size,-1).to(self.device)
        action = torch.cat(action).reshape(batch_size,-1).to(self.device)
        reward = torch.cat(reward).reshape(batch_size,-1).to(self.device)
        state_next = torch.cat(state_next).reshape(batch_size,-1).to(self.device)
        done = torch.cat(done).reshape(batch_size,-1).to(self.device)

        return state, action, reward, state_next, done
