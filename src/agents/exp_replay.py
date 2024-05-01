import torch
import numpy as np
from importlib import import_module
from .default import NormalRankNN
from .regularization import SI, L2, EWC, MAS
from ..dataloaders.wrapper import Storage
import gc


class Naive_Rehearsal(NormalRankNN):
    def __init__(self, agent_config):
        super(Naive_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 1000
        self.task_memory = {}
        self.skip_memory_concatenation = False

    def learn_batch(self, train_loader, val_loader=None):
        # 1.Combine training set
        if not train_loader:
            super(Naive_Rehearsal, self).learn_batch(None, val_loader)
            return
        if self.skip_memory_concatenation:
            new_train_loader = train_loader
        else: # default
            dataset_list = []
            for storage in self.task_memory.values():
                dataset_list.append(storage)
            dataset_list *= max(len(train_loader.dataset)//(self.memory_size), 1)  # Let old data: new data = 1:1
            dataset_list.append(train_loader.dataset)
            dataset = torch.utils.data.ConcatDataset(dataset_list)
            new_train_loader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=train_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=train_loader.num_workers)
            # 2.Update model as normal
            super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)
            del new_train_loader, dataset_list
            gc.collect()

            # 3.Randomly decide the images to stay in the memory
            self.task_count += 1
            # (a) Decide the number of samples for being saved
            num_sample_per_task = self.memory_size // self.task_count
            num_sample_per_task = min(len(train_loader.dataset), num_sample_per_task)
            # (b) Reduce current exemplar set to reserve the space for the new dataset
            for storage in self.task_memory.values():
                storage.reduce(num_sample_per_task)
            # (c) Randomly choose some samples from new task and save them to the memory
            randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
            self.task_memory[self.task_count] = Storage(train_loader.dataset, randind)


class Naive_Rehearsal_SI(Naive_Rehearsal, SI):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_SI, self).__init__(agent_config)


class Naive_Rehearsal_L2(Naive_Rehearsal, L2):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_L2, self).__init__(agent_config)


class Naive_Rehearsal_EWC(Naive_Rehearsal, EWC):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_EWC, self).__init__(agent_config)
        self.online_reg = True  # Online EWC


class Naive_Rehearsal_MAS(Naive_Rehearsal, MAS):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_MAS, self).__init__(agent_config)


class GEM(Naive_Rehearsal):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.quadprog = import_module('quadprog')
        self.task_mem_cache = {}

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project2cone2(self, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        margin = self.config['reg_coef']
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        #print(memories_np.shape, gradient_np.shape)
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = self.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    def learn_batch(self, train_loader, val_loader=None):

        # Update model as normal
        super(GEM, self).learn_batch(train_loader, val_loader)
        # Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=0)
            assert len(mem_loader) == 1, 'The length of mem_loader should be 1'
            for i, (mem_q, mem_pos, mem_neg) in enumerate(mem_loader):
                if self.gpu:
                    for idx, item in mem_q.items():
                        mem_q[idx] = item.cuda()
                    for idx, item in mem_pos.items():
                        mem_pos[idx] = item.cuda()
                    for idx, item in mem_neg.items():
                        mem_neg[idx] = item.cuda()
            self.task_mem_cache[t] = {'data': {'q': mem_q, 'pos': mem_pos, 'neg': mem_neg}}

    def update_model(self, q, pos, neg):

        # compute gradient on previous tasks
        if self.task_count > 0:
            for t, mem in self.task_memory.items():
                self.zero_grad()
                mem_q = self.task_mem_cache[t]['data']['q']
                mem_pos = self.task_mem_cache[t]['data']['pos']
                mem_neg = self.task_mem_cache[t]['data']['neg']
                mem_score_pos = self.forward(mem_q, mem_pos)
                mem_score_neg = self.forward(mem_q, mem_neg)
                mem_loss = self.criterion(mem_score_pos, mem_score_neg)
                mem_loss.backward()
                # Store the grads
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on the current minibatch
        score_pos = self.forward(q, pos)
        score_neg = self.forward(q, neg)
        loss = self.criterion(score_pos, score_neg)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach(), score_pos, score_neg
