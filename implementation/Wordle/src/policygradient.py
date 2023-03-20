import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import numpy as np
import gym


import wordle_gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNet(nn.Module):
    def __init__(self) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(30, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))
    
if __name__ ==  "__main__":
    num_episode = 5000
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.99
    
    env = gym.make('Wordle-v0')
    policy_net = PolicyNet()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    
    for e in range(num_episode):
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        env.render()
        
        for t in count():
            probs = policy_net(state)
            m = Bernoulli(probs)
            action = m.sample().data.numpy().astype(int)[0]
            next_state, reward, done, _ = env.step(action)
            
            if done:
                reward = 1200
            
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            
            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)
            
            steps +=1
            
            if done:
                break
        if e > 0 and e % batch_size == 0:
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i]-reward_mean)/reward_std
            
            optimizer.zero_grad()
            
            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]
                probs = policy_net(state)
                m = Bernoulli(probs)
                loss = -m.log_prob(action) * reward
                loss.backward()
            optimizer.step()
            
            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
            