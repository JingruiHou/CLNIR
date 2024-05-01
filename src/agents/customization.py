import torch
# from .default import NormalNN
from .regularization import SI, EWC, EWC_online
from .exp_replay import Naive_Rehearsal, GEM



def EWC_instance(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 6000000
    return agent


def EWC_online_instance(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 6000000
    agent.online_reg = True
    return agent


def EWC_online_empFI(agent_config):
    agent = EWC(agent_config)
    agent.empFI = True
    return agent


def EWC_rand_init(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    return agent


def EWC_reset_optim(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    return agent


def EWC_online_reset_optim(agent_config):
    agent = EWC_online(agent_config)
    agent.reset_optimizer = True
    return agent


def Naive_Rehearsal_100(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 100
    return agent


def Naive_Rehearsal_200(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 200
    return agent


def Naive_Rehearsal_400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 400
    return agent


def Naive_Rehearsal_1100(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1100
    return agent


def Naive_Rehearsal_1400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1400
    return agent


def Naive_Rehearsal_4000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4000
    return agent


def Naive_Rehearsal_4400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4400
    return agent


def Naive_Rehearsal_5600(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 5600
    return agent


def Naive_Rehearsal_16000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 16000
    return agent


def Naive_Rehearsal_6000000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 6000000
    return agent


def Naive_Rehearsal_4000000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4000000
    return agent

def Naive_Rehearsal_1000000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1000000
    return agent


def GEM_100(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 100
    return agent


def GEM_200(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 200
    return agent


def GEM_400(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 400
    return agent


def GEM_orig_1100(agent_config):
    agent = GEM(agent_config)
    agent.skip_memory_concatenation = True
    agent.memory_size = 1100
    return agent


def GEM_1100(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 1100
    return agent


def GEM_4000(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 4000
    return agent


def GEM_4400(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 4400
    return agent


def GEM_16000(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 16000
    return agent
