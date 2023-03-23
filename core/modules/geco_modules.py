import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from torch import linalg as la
from torch.distributions import normal

from core.utils.general_utils import ParamDict


class GECO(object):
    def __init__(self, params):
        self._hp = self._default_hparams().overwrite(params)
        self.cma = None
        self.geco_lambda = torch.tensor(self._hp.geco_lambda_init)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "alpha": 0.95,
                "geco_lambda_init": 1,
                "geco_lambda_min": 1e-10,
                "geco_lambda_max": 1e10,
                "speedup": None,
                "goal": None,
                "step_size": None,
            }
        )
        return default_dict

    def to(self, device):
        self.geco_lambda = self.geco_lambda.to(device)
        if self.cma is not None:
            self.cma = self.cma.to(device)

    def state_dict(self):
        return {"cma": self.cma, "geco_lambda": self.geco_lambda}

    def load_state_dict(self, state_dict):
        self.cma = state_dict["cma"]
        self.geco_lambda = state_dict["geco_lambda"]

    def loss(self, err, kld):
        loss = err + self.geco_lambda * kld

        with torch.no_grad():
            if self.cma is None:
                self.cma = err
            else:
                self.cma = (1.0 - self._hp.alpha) * err + self._hp.alpha * self.cma

            cons = self._hp.goal - self.cma
            if self._hp.speedup is not None and cons > 0:
                factor = torch.exp(self._hp.speedup * self._hp.step_size * cons)
            else:
                factor = torch.exp(self._hp.step_size * cons)
            self.geco_lambda = (factor * self.geco_lambda).clamp(
                self._hp.geco_lambda_min, self._hp.geco_lambda_max
            )

        return loss


class GECO1(object):
    def __init__(self, params):
        self._hp = self._default_hparams().overwrite(params)
        self.cma = None
        self.geco_lambda = torch.tensor(self._hp.geco_lambda_init)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "alpha": 0.95,
                "geco_lambda_init": 1,
                "geco_lambda_min": 1e-10,
                "geco_lambda_max": 1e10,
                "speedup": None,
                "goal": None,
                "step_size": None,
            }
        )
        return default_dict

    def to(self, device):
        self.geco_lambda = self.geco_lambda.to(device)
        if self.cma is not None:
            self.cma = self.cma.to(device)

    def state_dict(self):
        return {"cma": self.cma, "geco_lambda": self.geco_lambda}

    def load_state_dict(self, state_dict):
        self.cma = state_dict["cma"]
        self.geco_lambda = state_dict["geco_lambda"]

    def reset(self):
        self.cma = None
        self.geco_lambda = torch.tensor(self._hp.geco_lambda_init)

    def loss(self, err0, err1):
        constraint = err1 - self._hp.goal

        if self.cma is None:
            self.cma = constraint
        else:
            self.cma = (1.0 - self._hp.alpha) * constraint + self._hp.alpha * self.cma

        with torch.no_grad():
            factor = torch.exp(self._hp.step_size * self.cma)
            self.geco_lambda = (factor * self.geco_lambda).clamp(
                self._hp.geco_lambda_min, self._hp.geco_lambda_max
            )

        # print('Lambda: {}  err0: {}  constraint {}'.format(self.geco_lambda, err0, constraint))
        loss = err0 + self.geco_lambda * err1
        return loss


class GECO2(object):
    def __init__(self, params):
        self._hp = self._default_hparams().overwrite(params)
        self.cma1 = None
        self.cma2 = None
        self.geco_lambda1 = torch.tensor(self._hp.geco_lambda1_init)
        self.geco_lambda2 = torch.tensor(self._hp.geco_lambda2_init)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "alpha1": 0.95,
                "alpha2": 0.95,
                "geco_lambda1_init": 1,
                "geco_lambda2_init": 1,
                "geco_lambda1_min": 1e-10,
                "geco_lambda2_min": 1e-10,
                "geco_lambda1_max": 1e10,
                "geco_lambda2_max": 1e10,
                "speedup": None,
                "goal1": None,
                "goal2": None,
                "step_size1": None,
                "step_size2": None,
            }
        )
        return default_dict

    def to(self, device):
        self.geco_lambda1 = self.geco_lambda1.to(device)
        self.geco_lambda2 = self.geco_lambda2.to(device)
        if self.cma1 is not None:
            self.cma1 = self.cma1.to(device)
        if self.cma2 is not None:
            self.cma2 = self.cma2.to(device)

    def state_dict(self):
        return {
            "cma1": self.cma1.detach() if self.cma1 is not None else self.cma1,
            "cma2": self.cma2.detach() if self.cma2 is not None else self.cma2,
            "geco_lambda1": self.geco_lambda1,
            "geco_lambda2": self.geco_lambda2,
        }

    def load_state_dict(self, state_dict):
        self.cma1 = state_dict["cma1"]
        self.cma2 = state_dict["cma2"]
        self.geco_lambda1 = state_dict["geco_lambda1"]
        self.geco_lambda2 = state_dict["geco_lambda2"]

    def loss(self, err0, err1, err2):
        constraint1 = err1 - self._hp.goal1
        constraint2 = err2 - self._hp.goal2

        with torch.no_grad():
            if self.cma1 is None:
                self.cma1 = constraint1
            else:
                self.cma1 = (
                    1.0 - self._hp.alpha1
                ) * constraint1 + self._hp.alpha1 * self.cma1

            if self.cma2 is None:
                self.cma2 = constraint2
            else:
                self.cma2 = (
                    1.0 - self._hp.alpha2
                ) * constraint2 + self._hp.alpha2 * self.cma2

            factor1 = torch.exp(self._hp.step_size1 * self.cma1)
            self.geco_lambda1 = (factor1 * self.geco_lambda1).clamp(
                self._hp.geco_lambda1_min, self._hp.geco_lambda1_max
            )
            factor2 = torch.exp(self._hp.step_size2 * self.cma2)
            self.geco_lambda2 = (factor2 * self.geco_lambda2).clamp(
                self._hp.geco_lambda2_min, self._hp.geco_lambda2_max
            )

        # print('lambda1: ', self.geco_lambda1, ' lambda2: ', self.geco_lambda2)
        # print('loss0: ', err0, ' loss 1: ', self.geco_lambda1*err1, ' loss 2: ', self.geco_lambda2*err2)
        # print(err2)
        # loss = err0 + self.geco_lambda1 * err1 + self.geco_lambda2 * err2
        loss = err0 + self.geco_lambda1 * constraint1 + self.geco_lambda2 * constraint2

        return loss


class GECO3(object):
    def __init__(self, params):
        self._hp = self._default_hparams().overwrite(params)
        self.cma1 = None
        self.cma2 = None
        self.cma3 = None
        self.geco_lambda1 = torch.tensor(self._hp.geco_lambda1_init)
        self.geco_lambda2 = torch.tensor(self._hp.geco_lambda2_init)
        self.geco_lambda3 = torch.tensor(self._hp.geco_lambda3_init)

    def reset(self):
        self.cma1 = None
        self.cma2 = None
        self.cma3 = None
        self.geco_lambda1 = torch.tensor(self._hp.geco_lambda1_init)
        self.geco_lambda2 = torch.tensor(self._hp.geco_lambda2_init)
        self.geco_lambda3 = torch.tensor(self._hp.geco_lambda3_init)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "alpha1": 0.95,
                "alpha2": 0.95,
                "alpha3": 0.95,
                "geco_lambda1_init": 1,
                "geco_lambda2_init": 1,
                "geco_lambda3_init": 1,
                "geco_lambda1_min": 1e-10,
                "geco_lambda2_min": 1e-10,
                "geco_lambda3_min": 1e-10,
                "geco_lambda1_max": 1e10,
                "geco_lambda2_max": 1e10,
                "geco_lambda3_max": 1e10,
                "speedup": None,
                "goal1": None,
                "goal2": None,
                "goal3": None,
                "step_size1": None,
                "step_size2": None,
                "step_size3": None,
            }
        )
        return default_dict

    def to(self, device):
        self.geco_lambda1 = self.geco_lambda1.to(device)
        self.geco_lambda2 = self.geco_lambda2.to(device)
        self.geco_lambda3 = self.geco_lambda3.to(device)
        if self.cma1 is not None:
            self.cma1 = self.cma1.to(device)
        if self.cma2 is not None:
            self.cma2 = self.cma2.to(device)
        if self.cma3 is not None:
            self.cma3 = self.cma3.to(device)

    def state_dict(self):
        return {
            "cma1": self.cma1.detach() if self.cma1 is not None else self.cma1,
            "cma2": self.cma2.detach() if self.cma2 is not None else self.cma2,
            "cma3": self.cma3.detach() if self.cma3 is not None else self.cma3,
            "geco_lambda1": self.geco_lambda1,
            "geco_lambda2": self.geco_lambda2,
            "geco_lambda3": self.geco_lambda3,
        }

    def load_state_dict(self, state_dict):
        self.cma1 = state_dict["cma1"]
        self.cma2 = state_dict["cma2"]
        self.cma3 = state_dict["cma3"]
        self.geco_lambda1 = state_dict["geco_lambda1"]
        self.geco_lambda2 = state_dict["geco_lambda2"]
        self.geco_lambda3 = state_dict["geco_lambda3"]

    def loss(self, err0, err1, err2, err3):
        constraint1 = err1 - self._hp.goal1
        constraint2 = err2 - self._hp.goal2
        constraint3 = err3 - self._hp.goal3

        with torch.no_grad():
            if self.cma1 is None:
                self.cma1 = constraint1
            else:
                self.cma1 = (
                    1.0 - self._hp.alpha1
                ) * constraint1 + self._hp.alpha1 * self.cma1

            if self.cma2 is None:
                self.cma2 = constraint2
            else:
                self.cma2 = (
                    1.0 - self._hp.alpha2
                ) * constraint2 + self._hp.alpha2 * self.cma2

            if self.cma3 is None:
                self.cma3 = constraint3
            else:
                self.cma3 = (
                    1.0 - self._hp.alpha3
                ) * constraint3 + self._hp.alpha3 * self.cma3

            factor1 = torch.exp(self._hp.step_size1 * self.cma1)
            # if self.geco_lambda1 < 0:
            #     import pdb
            #     pdb.set_trace()
            self.geco_lambda1 = (factor1 * self.geco_lambda1).clamp(
                self._hp.geco_lambda1_min, self._hp.geco_lambda1_max
            )
            factor2 = torch.exp(self._hp.step_size2 * self.cma2)
            self.geco_lambda2 = (factor2 * self.geco_lambda2).clamp(
                self._hp.geco_lambda2_min, self._hp.geco_lambda2_max
            )
            factor3 = torch.exp(self._hp.step_size3 * self.cma3)
            self.geco_lambda3 = (factor3 * self.geco_lambda3).clamp(
                self._hp.geco_lambda3_min, self._hp.geco_lambda3_max
            )

        # loss = err0 + self.geco_lambda1 * constraint1 + self.geco_lambda2 * constraint2 + self.geco_lambda3 * constraint3
        loss = (
            err0
            + self.geco_lambda1 * err1
            + self.geco_lambda2 * err2
            + self.geco_lambda3 * err3
        )
        # print('lambda 1: {}, lambda 2: {}, lambda 3: {}'.format(self.geco_lambda1, self.geco_lambda2, self.geco_lambda3))
        # print('loss 1: {}, loss 2: {}, loss 3: {}'.format(self.geco_lambda1*err1, self.geco_lambda2*err2, self.geco_lambda3*err3))
        # if err3 > 0.25:
        #     # loss = err0 + self.geco_lambda3 * constraint3
        #     loss = err0 + self.geco_lambda3 * err3
        # else:
        #     loss = err0 + self.geco_lambda1 * err1 + self.geco_lambda2 * err2 + self.geco_lambda3 * err3
        # loss = err0 + self.geco_lambda1 * constraint1 + self.geco_lambda2 * constraint2 + self.geco_lambda3 * constraint3

        return loss
