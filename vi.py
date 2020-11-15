import math
import random
from math import nan

from pyro.distributions.transforms import CorrLCholeskyTransform
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.distributions
import pyro
from pyro import distributions as dist, poutine
from pyro.distributions import constraints
from pyro.infer import SVI as SVI_, Trace_ELBO, config_enumerate, TraceEnum_ELBO, Predictive, Importance
from pyro.infer.util import torch_item
from pyro.optim import Adam, PyroLRScheduler
from pyro.poutine.messenger import Messenger
from torch import nn
from torch.distributions import LowerCholeskyTransform
from tqdm import trange

# ======心理测量模型 start=============


def irt_1pl(x, b, D=1):
    """
    单参数IRT模型
    :param x: 潜变量
    :param b: 截距
    :return: 反应概率
    """
    return torch.sigmoid(D * (x + b))


def irt_2pl(x, a, b, D=1):
    """
    双参数IRT模型
    :param D: D
    :param x: 潜变量
    :param a: 斜率
    :param b: 截距
    :return: 反应概率
    """
    return torch.sigmoid(D * (x.mm(a) + b))


def irt_3pl(x, a, b, c, D=1):
    """
    三参数IRT模型
    :param x: 潜变量
    :param a: 斜率
    :param b: 截距
    :param c: 猜测参数
    :return: 反应概率
    """
    return c + (1 - c) * irt_2pl(x, a, b, D)


def irt_4pl(x, a, b, c, d, D=1):
    """
    四参数IRT模型
    :param x: 潜变量
    :param a: 斜率
    :param b: 截距
    :param c: 猜测参数
    :param d: 手滑参数
    :return: 反应概率
    """
    return c + (d - c) * irt_2pl(x, a, b, D)


def dina(attr, q, g, s):
    """
    DINA模型
    :param attr: 属性掌握模式
    :param q: Q矩阵
    :param g: 猜测参数
    :param s: 手滑参数
    :return:反应概率
    """
    yita = attr.mm(q)
    qq = (q ** 2).sum(dim=0)
    yita[yita < qq] = 0
    yita[yita == qq] = 1
    p = (1 - s) ** yita * g ** (1 - yita)
    return p


def ho_dina(lam0, lam1, theta, q, g, s):
    """
    HO-DINA模型
    :param lam0: 截距
    :param lam1: 斜率
    :param theta: 潜变量
    :param q: Q矩阵
    :param g: 猜测参数
    :param s: 手滑参数
    :return: 反应概率
    """
    attr_p = torch.sigmoid(theta.mm(lam1) + lam0)
    attr = dist.Bernoulli(attr_p).sample()
    return dina(attr, q, g, s)

# ======心理测量模型 end=============

# ======随机数据生成 start=======


class RandomMissing(object):

    def __init__(self, missing_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing_rate = missing_rate

    @property
    def y(self):
        _y = super().y
        y_size = _y.size(0) * _y.size(1)
        row_idx = torch.arange(0, _y.size(0)).repeat(int(_y.size(1) * self.missing_rate))
        col_idx = torch.randint(0, _y.size(1), (int(y_size * self.missing_rate),))
        _y[row_idx, col_idx] = nan
        return _y


class RandomPsyData(object):

    def __init__(self, sample_size=10000, item_size=100, *args, **kwargs):
        """
        :param sample_size: 样本量
        :param item_size: 题量
        """
        self.item_size = item_size
        self.sample_size = sample_size


class RandomDina(RandomPsyData):
    # 生成随机DINA模型数据
    name = 'dina'

    def __init__(self, q_size=5, q_p=0.5, attr_p=0.5, *args, **kwargs):
        """
        :param q_size: q矩阵的行数
        :param q_p: q矩阵的二项分布概率
        :param attr_p: 属性掌握的二项分布概率
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.q_size = q_size
        self.q = torch.FloatTensor(q_size, self.item_size).bernoulli_(q_p)
        q_eye = torch.eye(q_size)
        q_sum = self.q.sum(0)
        if torch.any(q_sum == 0):
            idx = dist.Categorical(torch.zeros(q_size) + 1).sample((self.q[:, q_sum == 0].size(1),))
            self.q[:, q_sum == 0] = q_eye[idx].T
        self.attr = torch.FloatTensor(self.sample_size, q_size).bernoulli_(attr_p)
        self.g = torch.FloatTensor(1, self.item_size).uniform_(0, 0.3)
        self.s = torch.FloatTensor(1, self.item_size).uniform_(0, 0.3)

    @property
    def y(self):
        p = dina(self.attr, self.q, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomLargeScaleDina(RandomPsyData):
    # 生成随机DINA模型数据
    name = 'dina'

    def __init__(self, q_size=5, corr=0.25, g_lower=0, g_upper=0.3, s_lower=0, s_upper=0.3, *args, **kwargs):
        """
        :param q_size: q矩阵的行数
        :param q_p: q矩阵的二项分布概率
        :param attr_p: 属性掌握的二项分布概率
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.q_size = q_size
        q_eye = torch.eye(q_size)
        q1 = torch.eye(q_size)
        q2 = torch.eye(q_size)
        for i in range(q_size):
            if i < q_size - 1:
                q1[i, i + 1] = 1
                q2[i, i + 1] = 1
            if i > 0:
                q2[i, i - 1] = 1
        self.q = torch.cat([q_eye, q1, q2]).T
        attr_mean = torch.zeros((q_size, ))
        attr_cov = torch.ones((q_size, q_size)) * corr
        attr_cov += torch.eye(q_size) * (1 - corr)
        attr = dist.MultivariateNormal(attr_mean, covariance_matrix=attr_cov).sample((self.sample_size, ))
        attr[attr > 0] = 1
        attr[attr <= 0] = 0
        self.attr = attr
        self.g = torch.FloatTensor(1, self.q_size * 3).uniform_(g_lower, g_upper)
        self.s = torch.FloatTensor(1, self.q_size * 3).uniform_(s_lower, s_upper)

    @property
    def y(self):
        p = dina(self.attr, self.q, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomHoDina(RandomDina):
    # 生成随机HO-DINA模型数据
    name = 'ho_dina'

    def __init__(self, theta_local=0, theta_scale=1, lam0_local=0, lam0_scale=1, lam1_lower=0.5,
                 lam1_upper=3, theta_feature=1,*args, **kwargs):
        """
        :param theta_dim: 潜变量维度
        :param theta_local: 潜变量正态分布的均值
        :param theta_scale: 潜变量正态分布的标准差
        :param lam0_local: 截距正态分布的均值
        :param lam0_scale: 截距正态分布的标准差
        :param lam1_lower: 斜率均匀分布的下界
        :param lam1_upper: 斜率均匀分布的上界
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.theta = torch.FloatTensor(self.sample_size, theta_feature).normal_(theta_local, theta_scale)
        self.lam0 = torch.FloatTensor(1, self.q_size).normal_(lam0_local, lam0_scale)
        self.lam1 = torch.FloatTensor(theta_feature, self.q_size).uniform_(lam1_lower, lam1_upper)
        for i in range(theta_feature):
            self.lam1[i, self.q_size - i:] = 0

    @property
    def y(self):
        p = ho_dina(self.lam0, self.lam1, self.theta, self.q, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomMissingHoDina(RandomMissing, RandomHoDina):
    pass


class RandomIrt1PL(RandomPsyData):
    # 生成随机单参数IRT模型数据
    name = 'irt_1pl'

    def __init__(
            self,
            x_feature=1,
            x_local=0,
            x_scale=1,
            x_cov=None,
            b_local=0,
            b_scale=1,
            D=1,
            *args,
            **kwargs
    ):
        """
        :param x_feature: 潜变量维度
        :param x_local: 潜变量正态分布的均值
        :param x_scale: 潜变量正态分布的标准差
        :param b_local: 截距正态分布的均值
        :param b_scale: 截距正态分布的标准差
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.x_feature = x_feature
        if x_cov is None:
            self.x = torch.FloatTensor(self.sample_size, x_feature).normal_(x_local, x_scale)
        else:
            self.x = dist.MultivariateNormal(x_local, scale_tril=x_cov).sample((self.sample_size,))
        self.b = torch.FloatTensor(1, self.item_size).normal_(b_local, b_scale)
        self.D = D

    @property
    def y(self):
        p = irt_1pl(self.x, self.b, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt2PL(RandomIrt1PL):
    # 生成随机双参数IRT模型数据
    name = 'irt_2pl'

    def __init__(
            self,
            a_lower=0.5,
            a_upper=3,
            *args,
            **kwargs
    ):
        """
        :param a_lower: 斜率均匀分布的上界
        :param a_upper: 斜率均匀分布的下界
        :param args:
        :param kwargs:
        """
        super(RandomIrt2PL, self).__init__(*args, **kwargs)
        self.a = torch.FloatTensor(self.x_feature, self.item_size).uniform_(a_lower, a_upper)
        for i in range(self.x_feature):
            self.a[i, self.item_size - i:] = 0


    @property
    def y(self):
        p = irt_2pl(self.x, self.a, self.b, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomMissingIrt2PL(RandomMissing, RandomIrt2PL):
    pass


class RandomIrt3PL(RandomIrt2PL):
    # 生成随机三参数IRT模型数据
    name = 'irt_3pl'

    def __init__(self, c_unif_lower=0.05, c_unif_upper=0.2, *args, **kwargs):
        """
        :param c_unif_lower: 猜测参数均匀分布的下界
        :param c_unif_upper: 猜测参数均匀分布的上界
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.c = torch.FloatTensor(1, self.item_size).uniform_(c_unif_lower, c_unif_upper)

    @property
    def y(self):
        p = irt_3pl(self.x, self.a, self.b, self.c, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt4PL(RandomIrt3PL):
    # 生成随机四参数IRT模型数据
    name = 'irt_4pl'

    def __init__(self, d_unif_lower=0.8, d_unif_upper=0.95, *args, **kwargs):
        """
        :param d_unif_lower: 手滑参数均匀分布的下界
        :param d_unif_upper: 手滑参数均匀分布的上界
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.d = torch.FloatTensor(1, self.item_size).uniform_(d_unif_lower, d_unif_upper)

    @property
    def y(self):
        p = irt_4pl(self.x, self.a, self.b, self.c, self.d, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomMilIrt2PL(RandomPsyData):

    name = 'irt_2pl'

    def __init__(
            self,
            mdisc_log_local=0,
            mdisc_log_scale=0.5,
            mdiff_local=0.5,
            mdiff_scale=1,
            x_feature=2,
            x_local=None,
            x_cov=None,
            D=1,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        mdisc = torch.FloatTensor(self.item_size).log_normal_(mdisc_log_local, mdisc_log_scale)
        mdiff = torch.FloatTensor(self.item_size).normal_(mdiff_local, mdiff_scale)
        self.a = self.gen_a(self.item_size, mdisc, x_feature)
        b = -mdiff * mdisc
        self.b = b.view(1, -1)
        self.x_feature = x_feature
        if x_local is None:
            x_local = torch.zeros((x_feature,))
        if x_cov is None:
            x_cov = torch.eye(x_feature)
        self.x = dist.MultivariateNormal(x_local, x_cov).sample((self.sample_size,))
        self.D = D

    @property
    def y(self):
        p = irt_2pl(self.x, self.a, self.b, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)

    @staticmethod
    def generate_randval(x_low, x_up, x_sum, y):
        # https://blog.csdn.net/maintony/article/details/88540320
        if len(x_low) == 1:
            y.append(x_sum)
        else:
            a = max(x_sum - sum(x_up[1:len(x_up)]), x_low[0])
            b = min(x_sum - sum(x_low[1:len(x_low)]), x_up[0])
            temp = random.uniform(a, b)
            y.append(temp)
            x_low = x_low[1:len(x_low)]
            x_up = x_up[1:len(x_up)]
            x_sum = x_sum - temp
            RandomMilIrt2PL.generate_randval(x_low, x_up, x_sum, y)

    def gen_omega(self, x_feature):
        x_low = [0 for _ in range(x_feature)]
        x_up = [np.pi / 2 for _ in range(x_feature)]
        x_sum = np.pi / 2 * (x_feature - 1)
        omega = []
        self.generate_randval(x_low, x_up, x_sum, omega)
        return omega

    def gen_a(self, item_size, mdisc, x_feature):
        a = torch.zeros((x_feature, item_size))
        for j in range(item_size):
            if j < item_size - x_feature + 1:
                sigma = self.gen_omega(x_feature)
                a[:, j] = mdisc[j] * torch.cos(torch.FloatTensor(sigma))
                a[a < 0.01] = 0.01
            else:
                x_size = item_size - j
                sigma = self.gen_omega(x_size)
                a[:x_size, j] = mdisc[j] * torch.cos(torch.FloatTensor(sigma))
                a[a < 0.01] = 0.01
        for i in range(x_feature):
            a[i, item_size - i:] = 0
        return a


class RandomMilIrt3PL(RandomMilIrt2PL):

    name = 'irt_3pl'

    def __init__(self, logit_c_local=-1.39, logit_c_scale=0.16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logit_c = torch.FloatTensor(1, self.item_size).normal_(logit_c_local, logit_c_scale)
        self.c = 1 / (1 + torch.exp(-logit_c))

    @property
    def y(self):
        p = irt_3pl(self.x, self.a, self.b, self.c, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomMilIrt4PL(RandomMilIrt3PL):

    name = 'irt_4pl'

    def __init__(self, logit_d_local=-1.39, logit_d_scale=0.16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logit_d = torch.FloatTensor(1, self.item_size).normal_(logit_d_local, logit_d_scale)
        self.d = 1 / (1 + torch.exp(logit_d))

    @property
    def y(self):
        p = irt_4pl(self.x, self.a, self.b, self.c, self.d, self.D)
        return torch.FloatTensor(*p.size()).bernoulli_(p)

# ======随机数据生成 end===============

# ======深度生成模型 start=============


class NormEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim):
        """
        :param item_size: 题量
        :param x_dim: 潜变量特质维度数
        :param hidden_dim: 隐藏层维度数
        """
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.fc22 = nn.Linear(hidden_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        x_loc = self.fc21(hidden)
        x_scale = torch.exp(self.fc22(hidden))
        return x_loc, x_scale


class MvnEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim, share_cov=False):
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.fc22 = nn.Linear(hidden_dim, int(x_dim * (x_dim + 1) / 2))
        self.x_dim = x_dim
        self.softplus = nn.Softplus()
        self.transform = LowerCholeskyTransform()
        self.share_cov = share_cov

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        x_loc = self.fc21(hidden)
        x_scale_ = self.fc22(hidden)
        idx = torch.tril_indices(self.x_dim, self.x_dim)
        if self.share_cov:
            x_scale = torch.zeros((self.x_dim, self.x_dim))
            x_scale[idx[0], idx[1]] = x_scale_.mean(0)
        else:
            x_scale = torch.zeros(x_scale_.shape[:-1] + (self.x_dim, self.x_dim))
            x_scale[..., idx[0], idx[1]] = x_scale_[..., :]
        return x_loc, self.transform(x_scale)


class PriorScaleEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(x_dim * (x_dim - 1) / 2))
        self.x_dim = x_dim
        self.softplus = nn.Softplus()
        self.transform = CorrLCholeskyTransform()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        x_scale_ = self.fc2(hidden)
        return self.transform(x_scale_.mean(0))


class BinEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        p = self.sigmoid(self.fc2(hidden))
        return p


class SoftmaxEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        p = self.softmax(self.fc2(hidden))
        return p

# ======深度生成模型 end=============

# ======参数约束 start==============


class FreeMessenger(Messenger):
    # 参数约束工具
    def __init__(self, free):
        super().__init__()
        self.free = free

    def _process_message(self, msg):
        msg["free"] = self.free if msg.get("free") is None else self.free & msg["free"]
        return None


class SVI(SVI_):
    # 加入了参数约束的svi
    def step(self, *args, **kwargs):
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)
        params = []
        for site in param_capture.trace.nodes.values():
            param = site["value"].unconstrained()
            if site.get('free') is not None:
                param.grad = site['free'] * param.grad
            params.append(param)
        self.optim(params)
        pyro.infer.util.zero_grads(params)
        return torch_item(loss)

# ======参数约束 end==============


class BasePsy(nn.Module):

    def __init__(self, data, subsample_size=None, **kwargs):
        """
        :param data: 作答反应矩阵
        :param subsample_size: mini-batch 样本数
        """
        super().__init__()
        self.data = data
        self.sample_size = data.size(0)
        self.item_size = data.size(1)
        self.subsample_size = subsample_size if subsample_size is not None else self.sample_size
        self.kwargs = kwargs


class BaseIRT(BasePsy):

    IRT_FUN = {
        'irt_1pl': irt_1pl,
        'irt_2pl': irt_2pl,
        'irt_3pl': irt_3pl,
        'irt_4pl': irt_4pl,
    }

    def __init__(self,
                 model='irt_2pl',
                 x_feature=1,
                 share_posterior_cov=False,
                 share_prior_cov=False,
                 neural_prior_cov=False,
                 D=1,
                 prior_free=False,
                 *args,
                 **kwargs
                 ):
        """
        :param model: irt模型，内容参考IRT_FUN键值
        :param x_feature: 潜变量特征维度数
        :param share_posterior_cov: 是否共享后验方差协方差矩阵
        :param D: irt模型的D值，一般是1.702或1
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self._model = model
        self.x_feature = x_feature
        self.share_posterior_cov = share_posterior_cov
        self.share_prior_cov = share_prior_cov
        self.neural_prior_cov = neural_prior_cov
        self.D = D
        self.prior_free = prior_free
        self.a_free = kwargs.get('a_free')
        self.a0 = kwargs.get('a0', torch.ones((self.x_feature, self.item_size)))
        if x_feature > 1 and self.a_free is None:
            self.a_free = torch.BoolTensor(x_feature, self.item_size).fill_(True)
            for i in range(x_feature):
                self.a_free[i, self.item_size - i:] = False
                self.a0[i, self.item_size - i:] = 0

    def get_prior_cov(self):
        raise NotImplementedError

    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        b0 = self.kwargs.get('b0', torch.zeros((1, self.item_size)))
        irt_param_kwargs = {'b': pyro.param('b', b0), 'D': self.D}
        if self._model in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
            with FreeMessenger(free=self.a_free):
                irt_param_kwargs['a'] = pyro.param('a', self.a0)
        if self._model in ('irt_3pl', 'irt_4pl'):
            irt_param_kwargs['c'] = pyro.param('c', torch.zeros((1, item_size)) + 0.1,
                                               constraint=constraints.unit_interval)
        if self._model == 'irt_4pl':
            irt_param_kwargs['d'] = pyro.param('d', torch.ones((1, item_size)) - 0.1,
                                               constraint=constraints.unit_interval)
        if self.x_feature == 1:
            with pyro.plate("data", sample_size, dim=-2) as idx:
                irt_param_kwargs['x'] = pyro.sample(
                    'x',
                    dist.Normal(torch.zeros((len(idx), self.x_feature)), torch.ones((len(idx), self.x_feature)))
                )
                p, data_ = self._get_p_data(data, idx, irt_param_kwargs)
                pyro.sample('y', dist.Bernoulli(p), obs=data_)
        else:
            x_cov0 = torch.eye(self.x_feature)
            if self.share_prior_cov:
                if self.prior_free:
                    if not self.neural_prior_cov:
                        x_cov = pyro.param('x_cov0', x_cov0, constraint=constraints.corr_cholesky_constraint)
                    else:
                        pyro.module('prior_encoder', self.prior_encoder)
                else:
                    x_cov = x_cov0
            else:
                if self.prior_free:
                    x_cov_ = pyro.param(
                        'x_cov',
                        x_cov0.repeat(sample_size, 1, 1),
                        constraint=constraints.corr_cholesky_constraint
                    )
                else:
                    x_cov_ = x_cov0.repeat(sample_size, 1, 1)
            with pyro.plate("data", sample_size) as idx:
                if self.share_prior_cov and self.neural_prior_cov:
                    x_cov = self.prior_encoder(self.data[idx])
                if not self.share_prior_cov:
                    x_cov = x_cov_[idx]
                irt_param_kwargs['x'] = pyro.sample(
                    'x',
                    dist.MultivariateNormal(
                        torch.zeros((len(idx), self.x_feature)),
                        scale_tril=x_cov
                    )
                )
                p, data_ = self._get_p_data(data, idx, irt_param_kwargs)
                pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data_)

    def _get_p_data(self, data, idx, irt_param_kwargs):
        irt_fun = self.IRT_FUN[self._model]
        p = irt_fun(**irt_param_kwargs)
        data_ = data[idx]
        data_nan = torch.isnan(data_)
        if data_nan.any():
            data_ = torch.where(data_nan, torch.full_like(data_, 0), data_)
            p = torch.where(data_nan, torch.full_like(p, 0), p)
        return p, data_

    def get_roc_auc(self, data):
        with torch.no_grad():
            data_nan = torch.isnan(data)
            if data_nan.any():
                data = torch.where(data_nan, torch.full_like(data, -1), data)
            num_posterior_samples = 1
            a = pyro.param('a')
            b = pyro.param('b')
            x_local, _ = self.encoder.forward(data)
            # if self.x_feature == 1:
            #     x = dist.Normal(x_local, x_scale).sample((num_posterior_samples,))
            # else:
            #     transform = LowerCholeskyTransform()
            #     x = dist.MultivariateNormal(x_local, scale_tril=transform(x_scale)).sample((num_posterior_samples,))
            y_pred = []
            for i in range(num_posterior_samples):
                y = irt_2pl(x_local, a, b)
                y_pred.append(y[data != -1])
            y_pred = torch.stack(y_pred).view(-1)
            y_true = data[data != -1].repeat(num_posterior_samples, 1).view(-1)
            roc_auc = roc_auc_score(
                y_true.numpy(),
                y_pred.numpy(),
            )
            return roc_auc

    def get_marginal(self, data):
        with torch.no_grad():
            posterior = Importance(
                model=self.model,
                guide=self.guide,
                num_samples=100,
            )
            posterior = posterior.run(data)
            log_weights = torch.stack(posterior.log_weights)
            marginal = torch.logsumexp(log_weights, 0) - math.log(log_weights.size(0))
        return marginal

    def fit(self, optim=Adam({'lr': 5e-2}), loss=Trace_ELBO(num_particles=1), max_iter=5000, random_instance=None):
        """
        :param optim: 优化算法
        :param loss: 损失函数
        :param max_iter: 最大迭代次数
        :param random_instance: 随机数据生成实例
        """
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)
        with trange(max_iter, disable=True) as t:
            for i in t:
                t.set_description(f'迭代：{i}')
                loss = svi.step(self.data)
                if isinstance(optim, PyroLRScheduler):
                    optim.step()
                if i % 3500 == 0:
                    # marginal = self.test().item()
                    # print(marginal)
                    val_data = self.kwargs.get('val_data', self.data)
                    roc_auc = self.get_roc_auc(val_data)
                    print(roc_auc)
                    roc_auc1 = self.get_roc_auc(self.data)
                    print(roc_auc1)
                with torch.no_grad():
                    postfix_kwargs = {}
                    if random_instance is not None:
                        b = pyro.param('b')
                        postfix_kwargs['threshold_error'] = '{0}'.format((b - random_instance.b).pow(2).sqrt().mean())
                        # x, _ = self.encoder.forward(self.data)
                        # x = pyro.param('x_local')
                        # postfix_kwargs['x_error'] = '{0}'.format((x - random_instance.x).pow(2).sqrt().mean())
                        # x_cov0 = pyro.param('x_cov0')
                        # x_cov = torch.eye(2)
                        # x_cov[0, 1] = x_cov[1, 0] = 0.7
                        # postfix_kwargs['x_cov0'] = '{0}'.format((x_cov0.mm(x_cov0.T) - x_cov).pow(2).sqrt().sum() / 2)
                        if self._model in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
                            a = pyro.param('a')
                            a_error = (a - random_instance.a).pow(2).sqrt().sum() / (self.x_feature * self.item_size - self.x_feature * (self.x_feature - 1) / 2)
                            postfix_kwargs['slop_error'] = '{0}'.format(a_error)
                        if self._model in ('irt_3pl', 'irt_4pl'):
                            c = pyro.param('c')
                            postfix_kwargs['guess_error'] = '{0}'.format((c - random_instance.c).pow(2).sqrt().mean())
                        if self._model == 'irt_4pl':
                            d = pyro.param('d')
                            postfix_kwargs['slip_error'] = '{0}'.format((d - random_instance.d).pow(2).sqrt().mean())
                    # t.set_postfix(loss='{0:1.2f}'.format(loss), **postfix_kwargs)


class VaeIRT(BaseIRT):

    # 基于变分自编码器的IRT参数估计
    def __init__(self, hidden_dim=64, neural_share_posterior_cov=False, *args, **kwargs):
        """
        :param hidden_dim: 隐藏层维度数
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if self.x_feature == 1:
            self.encoder = NormEncoder(self.item_size, self.x_feature, hidden_dim)
        else:
            self.encoder = MvnEncoder(self.item_size, self.x_feature, hidden_dim, self.share_posterior_cov)
            self.prior_encoder = PriorScaleEncoder(self.item_size, self.x_feature, 64)
            self.neural_share_posterior_cov = neural_share_posterior_cov

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        pyro.module('encoder', self.encoder)
        if self.x_feature == 1:
            with pyro.plate("data", sample_size, subsample_size=subsample_size, dim=-2) as idx:
                data_ = data[idx]
                data_nan = torch.isnan(data_)
                if data_nan.any():
                    data_ = torch.where(data_nan, torch.full_like(data_, -1), data_)
                x_local, x_scale = self.encoder.forward(data_)
                pyro.sample('x', dist.Normal(x_local, x_scale))
        else:
            if self.share_posterior_cov and not self.neural_share_posterior_cov:
                x_scale = pyro.param(
                    'x_scale',
                    torch.eye(self.x_feature),
                    constraint=constraints.lower_cholesky
                )
            with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
                data_ = data[idx]
                data_nan = torch.isnan(data_)
                if data_nan.any():
                    data_ = torch.where(data_nan, torch.full_like(data_, -1), data_)
                if self.share_posterior_cov and not self.neural_share_posterior_cov:
                    x_local, _ = self.encoder.forward(data_)
                else:
                    x_local, x_scale = self.encoder.forward(data_)
                pyro.sample('x', dist.MultivariateNormal(x_local, scale_tril=x_scale))



class VIRT(BaseIRT):
    # 基于黑盒变分推断的IRT参数估计
    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        if self.x_feature == 1:
            x_local = pyro.param('x_local', torch.zeros((sample_size, self.x_feature)))
            x_scale = pyro.param('x_scale', torch.ones((sample_size, self.x_feature)), constraint=constraints.positive)
            with pyro.plate("data", sample_size, subsample_size=subsample_size, dim=-2) as idx:
                pyro.sample('x', dist.Normal(x_local[idx], x_scale[idx]))
        else:
            x_local = pyro.param('x_local', torch.zeros((sample_size, self.x_feature)))
            if self.share_posterior_cov:
                x_scale_tril = pyro.param(
                    'x_scale',
                    torch.eye(self.x_feature),
                    constraint=constraints.lower_cholesky
                )
                with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
                    pyro.sample('x', dist.MultivariateNormal(x_local[idx], scale_tril=x_scale_tril))
            else:
                x_scale_tril = pyro.param(
                    'x_scale',
                    torch.eye(self.x_feature).repeat(sample_size, 1, 1),
                    constraint=constraints.lower_cholesky
                )
                with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
                    pyro.sample('x', dist.MultivariateNormal(x_local[idx], scale_tril=x_scale_tril[idx]))


class BaseCDM(BasePsy):

    CDM_FUN = {
        'dina': dina
    }

    def __init__(self, q, model='dina', *args, **kwargs):
        """
        :param q: q矩阵
        :param model: 认知诊断模型，参考CDM_FUN
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.q = q
        self.attr_size = q.size(0)
        self._model = model

    def fit(self, optim=Adam({'lr': 1e-3}), loss=Trace_ELBO(num_particles=1), max_iter=5000, random_instance=None):
        """
        :param optim: 优化算法
        :param loss: 损失函数
        :param max_iter: 最大迭代次数
        :param random_instance: 随机数据生成实例
        """
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)
        with trange(max_iter) as t:
            for i in t:
                t.set_description(f'迭代：{i}')
                svi.step(self.data)
                loss = svi.evaluate_loss(self.data)
                if isinstance(optim, PyroLRScheduler):
                    optim.step()
                with torch.no_grad():
                    postfix_kwargs = {}
                    if random_instance is not None:
                        g = pyro.param('g')
                        s = pyro.param('s')
                        postfix_kwargs.update({
                            'g': '{0}'.format((g - random_instance.g).abs().mean()),
                            's': '{0}'.format((s - random_instance.s).abs().mean())
                        })
                    t.set_postfix(loss=loss, **postfix_kwargs)


class VCDM(BaseCDM):
    # 基于离散潜变量黑盒变分推断的CDM参数估计，效果绝佳
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_attr = self.get_all_attrs()

    def get_all_attrs(self):
        row_size = 2 ** self.attr_size
        all_attr = torch.zeros((row_size, self.attr_size))
        for i in range(row_size):
            num = i
            count = 0
            while True:
                if num == 0:
                    break
                num, rem = divmod(num, 2)
                all_attr[i][count] = rem
                count += 1
        return all_attr

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        all_p = self.CDM_FUN[self._model](self.all_attr, self.q, g, s)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            attr_idx = pyro.sample(
                'attr_idx',
                dist.Categorical(torch.zeros((len(idx), self.all_attr.size(0))) + 1 / self.all_attr.size(0)).to_event(0)
            )
            p = all_p[attr_idx]
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, 0), data_)
                p = torch.where(data_nan, torch.full_like(p, 0), p)
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data_)

    def guide(self, data):
        pass

    def fit(self, loss=TraceEnum_ELBO(num_particles=1), *args, **kwargs):
        super().fit(loss=loss, *args, **kwargs)


class VaeCDM(VCDM):
    # 基于离散潜变量变分自编码器的CDM参数估计，效果绝佳
    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = SoftmaxEncoder(self.item_size, self.all_attr.size(0), hidden_dim)

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        pyro.module('encoder', self.encoder)
        all_p = self.CDM_FUN[self._model](self.all_attr, self.q, g, s)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, -1), data_)
            attr_p = self.encoder.forward(data_)
            attr_idx = pyro.sample(
                'attr_idx',
                dist.Categorical(attr_p).to_event(0)
            )
            p = all_p[attr_idx]
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data_)


class VHoDina(VCDM):
    # 基于离散潜变量黑盒变分推断的HO-DINA参数估计，效果绝佳
    def __init__(self, theta_feature=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_feature = theta_feature
        self.lam1_init = torch.ones((theta_feature, self.attr_size))
        self.lam1_free = torch.BoolTensor(theta_feature, self.attr_size).fill_(True)
        for i in range(theta_feature):
            self.lam1_free[i, self.attr_size - i:] = False
            self.lam1_init[i, self.attr_size - i:] = 0

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        lam0 = pyro.param('lam0', torch.zeros((1, self.attr_size)))
        with FreeMessenger(free=self.lam1_free):
            lam1 = pyro.param('lam1', self.lam1_init)
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        all_p = dina(self.all_attr, self.q, g, s)
        with pyro.plate("data", sample_size) as idx:
            if self.theta_feature < 2:
                theta = pyro.sample(
                    'theta',
                    dist.Normal(torch.zeros((len(idx), 1)), torch.ones((len(idx), 1))).to_event(1)
                )
            else:
                theta = pyro.sample(
                    'theta',
                    dist.MultivariateNormal(
                        torch.zeros((len(idx), self.theta_feature)),
                        torch.eye(self.theta_feature)
                    )
                )
            attr_p = torch.sigmoid(theta.mm(lam1) + lam0)
            likelihood_attr_p = torch.exp(torch.log(attr_p).mm(self.all_attr.T) + torch.log(1 - attr_p).mm(1 - self.all_attr.T))
            attr_idx = pyro.sample(
                'attr_idx',
                dist.Categorical(likelihood_attr_p).to_event(0)
            )
            p = all_p[attr_idx]
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, 0), data_)
                p = torch.where(data_nan, torch.full_like(p, 0), p)
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data_)

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        theta_local = pyro.param('theta_local', torch.zeros((sample_size, 1)))
        theta_scale = pyro.param('theta_scale', torch.ones((sample_size, 1)), constraint=constraints.positive)
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            pyro.sample(
                'theta',
                dist.Normal(theta_local[idx], theta_scale[idx]).to_event(1)
            )

    def fit(self, optim=Adam({'lr': 5e-3}), loss=TraceEnum_ELBO(num_particles=1), max_iter=50000, random_instance=None):
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)
        with trange(max_iter) as t:
            for i in t:
                t.set_description(f'迭代：{i}')
                svi.step(self.data)
                loss = svi.evaluate_loss(self.data)
                if isinstance(optim, PyroLRScheduler):
                    optim.step()
                with torch.no_grad():
                    postfix_kwargs = {}
                    if random_instance is not None:
                        g = pyro.param('g')
                        s = pyro.param('s')
                        lam0 = pyro.param('lam0')
                        lam1 = pyro.param('lam1')
                        postfix_kwargs.update({
                            'g': '{0}'.format((g - random_instance.g).pow(2).sqrt().mean()),
                            's': '{0}'.format((s - random_instance.s).pow(2).sqrt().mean()),
                            'lam0': '{0}'.format((lam0 - random_instance.lam0).pow(2).sqrt().mean()),
                            'lam1': '{0}'.format(
                                (lam1 - random_instance.lam1).pow(2).sqrt().sum()
                                /
                                (self.theta_feature * self.attr_size - self.theta_feature * (self.theta_feature - 1) / 2)),
                        })
                    t.set_postfix(loss=loss, **postfix_kwargs)


class VaeHoDina(VHoDina):
    # 基于离散潜变量变分自编码器的HO-DINA参数估计，效果绝佳

    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.theta_feature > 1:
            self.encoder = MvnEncoder(self.item_size, self.theta_feature, hidden_dim, share_cov=False)
        else:
            self.encoder = NormEncoder(self.item_size, 1, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, -1), data_)
            theta_local, theta_scale = self.encoder.forward(data_)
            if self.theta_feature > 1:
                pyro.sample(
                    'theta',
                    dist.MultivariateNormal(theta_local, scale_tril=theta_scale)
                )
            else:
                pyro.sample(
                    'theta',
                    dist.Normal(theta_local, theta_scale).to_event(1)
                )


class JABaseCDM(BaseCDM):
    # just another BaseCDM

    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        with pyro.plate("data", sample_size) as idx:
            attr = pyro.sample(
                'attr',
                dist.Bernoulli(torch.zeros((len(idx), self.attr_size)) + 0.5).to_event(1)
            )
            p = self.CDM_FUN[self._model](attr, self.q, g, s)
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, 0), data_)
                p = torch.where(data_nan, torch.full_like(p, 0), p)
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data_)

    def fit(self, optim=Adam({'lr': 1e-3}), loss=Trace_ELBO(num_particles=1), max_iter=5000, random_instance=None):
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)
        with trange(max_iter) as t:
            for i in t:
                t.set_description(f'迭代：{i}')
                svi.step(self.data)
                loss = svi.evaluate_loss(self.data)
                with torch.no_grad():
                    postfix_kwargs = {}
                    if random_instance is not None:
                        attr_p = self.encoder.forward(self.data)
                        # attr_p = pyro.param('attr_p')
                        attr_p[attr_p > 0.5] = 1
                        attr_p[attr_p <= 0.5] = 0
                        ac = attr_p - random_instance.attr
                        a_a = len(ac[ac == 0]) / (attr_p.size(0) * attr_p.size(1))
                        g = pyro.param('g')
                        s = pyro.param('s')
                        postfix_kwargs.update({
                            'g': '{0}'.format((g - random_instance.g).pow(2).sqrt().mean()),
                            's': '{0}'.format((s - random_instance.s).pow(2).sqrt().mean()),
                            'attr_p': '{0}'.format(a_a)
                        })
                    t.set_postfix(loss=loss, **postfix_kwargs)


class JAVaeCDM(JABaseCDM):
    # # just another 基于变分自编码器的CDM参数估计
    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = BinEncoder(self.item_size, self.attr_size, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, -1), data_)
            attr_p = self.encoder.forward(data_)
            pyro.sample(
                'attr',
                dist.Bernoulli(attr_p).to_event(1)
            )


class JAVCDM(JABaseCDM):
    # just another 基于黑盒变分推断的CDM参数估计
    def guide(self, data):
        sample_size = self.sample_size
        attr_p = pyro.param('attr_p', torch.zeros((sample_size, self.attr_size)) + 0.5, constraint=constraints.unit_interval)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            pyro.sample(
                'attr',
                dist.Bernoulli(attr_p[idx]).to_event(1)
            )