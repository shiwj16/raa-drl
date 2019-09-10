# -*- coding: utf-8 -*-

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RAA(object):
    def __init__(self, num_critics, use_restart, reg=0.1):
        self.size = num_critics
        self.reg = reg                 # regularization
        self.use_restart = use_restart
        self.count = 0
        self.interval = 5000
        self.errors = torch.zeros(self.interval).to(device)
        self.opt_error = torch.tensor(0.).to(device)

    def calculate(self, Qs, F_Qs):
        Qs = Qs.t()
        F_Qs = F_Qs.t()
        delta_Qs = F_Qs - Qs
        cur_size = Qs.size(1)

        del_mat = delta_Qs.t().mm(delta_Qs)
        alpha = del_mat / torch.abs(torch.mean(del_mat))
        alpha += self.reg * torch.eye(cur_size).to(device)

        alpha = torch.sum(alpha.inverse(), 1)
        alpha = torch.unsqueeze(alpha / torch.sum(alpha), 1)

        # restart checking
        self.count += 1
        self.errors[self.count % self.interval] = torch.mean(torch.pow(delta_Qs[:, -1], 2)).detach()

        if self.use_restart:
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

