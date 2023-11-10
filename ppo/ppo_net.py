
import torch
import torch.nn.functional as F
import numpy as np

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(state_dim,hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )

    def forward(self,x):
        x=self.net(x)
        log_softmax=F.log_softmax(x+1e-5,dim=-1)
        max_l=torch.max(log_softmax,dim=-1,keepdim=True)[0]
        log_softmax=torch.clip(log_softmax-max_l,-1e10,1)
        softmax=torch.exp(log_softmax)+1e-5
        probs=softmax/torch.sum(softmax,dim=-1, keepdim=True)

        return probs

class Net(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Net, self).__init__()

        self.state_embed=torch.nn.Sequential(
            torch.nn.Linear(state_dim,hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.value=torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,1)
        )
        self.actor=PolicyNet(hidden_dim,hidden_dim,action_dim)

    def get_state_embed(self,x):
        return self.state_embed(x)

    def forward(self,x):
        state_embed=self.get_state_embed(x)
        probs=self.actor(state_embed)
        value=self.value(state_embed)
        return probs,value
    def probs(self,x):
        state_embed=self.get_state_embed(x)
        return self.actor(state_embed)
    def values(self,x):
        state_embed=self.get_state_embed(x)
        return self.value(state_embed)