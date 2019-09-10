# -*- coding: utf-8 -*-
"""
Created on March 11 2019

@author: Wenjie Shi
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class raa(object):
    def __init__(self, num_critics, use_restart, reg=0.01):
        self.size = num_critics
        # regularization
        self.reg = reg
        self.use_restart = use_restart
        self.count = 0
        self.interval = 5000
        self.errors = torch.zeros(self.interval).to(device)
        self.opt_error = torch.tensor(0.).to(device)

    def calculate(self, Qs, F_Qs):
        Qs = Qs.squeeze(2).t()
        F_Qs = F_Qs.squeeze(2).t()
        delta_Qs = F_Qs - Qs
        cur_size = Qs.size(1)

        del_mat = delta_Qs.t().mm(delta_Qs)
        alpha = del_mat / torch.abs(torch.mean(del_mat))
        alpha += self.reg * torch.eye(cur_size).to(device)

        alpha = torch.sum(alpha.inverse(), 1)
        alpha = torch.unsqueeze(alpha / torch.sum(alpha), 1)

        # assert
        if self.use_restart:
            self.count += 1
            self.errors[self.count % self.interval] = torch.abs(torch.mean(delta_Qs[:, -1])).detach()

            if self.count % self.interval == 0:
                error = torch.mean(self.errors)
                if self.count == self.interval:
                    self.opt_error = error
                else:
                    self.opt_error = torch.min(self.opt_error, error)

                if (self.count > self.interval and error > self.opt_error) or self.count > 100000:
                    print(error, self.opt_error)
                    restart = True
                    self.count = 0
                else:
                    restart = False
            else:
                restart = False
        else:
            restart = False
    
        return alpha, restart

