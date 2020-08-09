from math import nan

import torch
import torch.distributions
import pyro
from pyro import distributions as dist, poutine
from pyro.distributions import constraints
from pyro.infer import SVI as SVI_, Trace_ELBO, config_enumerate, TraceEnum_ELBO, TraceMeanField_ELBO, TraceGraph_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from pyro.infer.util import torch_item
from pyro.optim import Adam
from pyro.poutine.messenger import Messenger
from torch import nn
from torch.distributions import LowerCholeskyTransform
from torch.nn import init
from tqdm import trange

# ======心理测量模型 start=============


def irt_1pl(x, b):
    """
    单参数IRT模型
    :param x: 潜变量
    :param b: 截距
    :return: 反应概率
    """
    return torch.sigmoid(x + b)


def irt_2pl(x, a, b):
    """
    双参数IRT模型
    :param x: 潜变量
    :param a: 斜率
    :param b: 截距
    :return: 反应概率
    """
    return torch.sigmoid(x.mm(a) + b)


def irt_3pl(x, a, b, c):
    """
    三参数IRT模型
    :param x: 潜变量
    :param a: 斜率
    :param b: 截距
    :param c: 猜测参数
    :return: 反应概率
    """
    return c + (1 - c) * irt_2pl(x, a, b)


def irt_4pl(x, a, b, c, d):
    """
    四参数IRT模型
    :param x: 潜变量
    :param a: 斜率
    :param b: 截距
    :param c: 猜测参数
    :param d: 手滑参数
    :return: 反应概率
    """
    return c + (d - c) * irt_2pl(x, a, b)


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
    aa = (q ** 2).sum(dim=0)
    yita[yita < aa] = 0
    yita[yita == aa] = 1
    p = (1 - s) ** yita * g ** (1 - yita)
    return p


def dino(attr, q, g, s):
    """
    DINO模型
    :param attr: 属性掌握模式
    :param q: Q矩阵
    :param g: 猜测参数
    :param s: 手滑参数
    :return: 反应概率
    """
    yita = (1 - attr).mm(q)
    aa = (q ** 2).sum(dim=0)
    yita[yita < aa] = 1
    yita[yita == aa] = 0
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
        self.attr = torch.FloatTensor(self.sample_size, q_size).bernoulli_(attr_p)
        self.g = torch.FloatTensor(1, self.item_size).uniform_(0, 0.3)
        self.s = torch.FloatTensor(1, self.item_size).uniform_(0, 0.3)

    @property
    def y(self):
        p = dina(self.attr, self.q, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomDino(RandomDina):
    # 生成随机DINO模型数据
    name = 'dino'

    @property
    def y(self):
        p = dino(self.attr, self.q, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomHoDina(RandomDina):
    # 生成随机HO-DINA模型数据
    name = 'ho_dina'

    def __init__(self, theta_dim=1, theta_local=0, theta_scale=1, lam0_local=0, lam0_scale=1, lam1_lower=0.5,
                 lam1_upper=3, *args, **kwargs):
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
        self.theta = torch.FloatTensor(self.sample_size, theta_dim).normal_(theta_local, theta_scale)
        self.lam0 = torch.FloatTensor(theta_dim, self.q_size).normal_(lam0_local, lam0_scale)
        self.lam1 = torch.FloatTensor(theta_dim, self.q_size).uniform_(lam1_lower, lam1_upper)

    @property
    def y(self):
        p = ho_dina(self.lam0, self.lam1, self.theta, self.q, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt1PL(RandomPsyData):
    # 生成随机单参数IRT模型数据
    name = 'irt_1pl'

    def __init__(
            self,
            x_feature=1,
            x_local=0,
            x_scale=1,
            b_local=0,
            b_scale=1,
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
        self.x = torch.FloatTensor(self.sample_size, x_feature).normal_(x_local, x_scale)
        self.b = torch.FloatTensor(1, self.item_size).normal_(b_local, b_scale)

    @property
    def y(self):
        p = irt_1pl(self.x, self.b)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt2PL(RandomIrt1PL):
    # 生成随机双参数IRT模型数据
    name = 'irt_2pl'

    def __init__(
            self,
            a_lower=1,
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

    @property
    def y(self):
        p = irt_2pl(self.x, self.a, self.b)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


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
        p = irt_3pl(self.x, self.a, self.b, self.c)
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
        p = irt_4pl(self.x, self.a, self.b, self.c, self.d)
        return torch.FloatTensor(*p.size()).bernoulli_(p)

# ======随机数据生成 end===============

# ======深度生成模型 start=============


class NormEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        self.fc22 = nn.Linear(hidden_dim, x_dim)
        self.softplus = nn.Softplus()
        self.transform = LowerCholeskyTransform()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        x_loc = self.fc21(hidden)
        s = self.fc22(hidden)
        s1 = s.unsqueeze(2)
        s2 = s.unsqueeze(1)
        cov = s1.bmm(s2)
        return x_loc, self.transform(cov)


class BinEncoder(nn.Module):

    def __init__(self, item_size, x_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(item_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
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
    def __init__(self, free):
        super().__init__()
        self.free = free

    def _process_message(self, msg):
        msg["free"] = self.free if msg.get("free") is None else self.free & msg["free"]
        return None


class SVI(SVI_):

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

    def __init__(self, model='irt_2pl', x_feature=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = model
        self.x_feature = x_feature
        self.a_free = kwargs.get('a_free')
        self.a0 = kwargs.get('a0', torch.ones((self.x_feature, self.item_size)))
        if x_feature > 1 and self.a_free is None:
            self.a_free = torch.BoolTensor(x_feature, self.item_size).fill_(True)
            for i in range(x_feature):
                self.a_free[i, self.item_size - i:] = False
                self.a0[i, self.item_size - i:] = 0

    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        b0 = self.kwargs.get('b0', torch.zeros((1, self.item_size)))
        irt_param_kwargs = {'b': pyro.param('b', b0)}
        if self._model in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
            with FreeMessenger(free=self.a_free):
                irt_param_kwargs['a'] = pyro.param('a', self.a0)
        if self._model in ('irt_3pl', 'irt_4pl'):
            irt_param_kwargs['c'] = pyro.param('c', torch.zeros((1, item_size)) + 0.1,
                                               constraint=constraints.unit_interval)
        if self._model == 'irt_4pl':
            irt_param_kwargs['d'] = pyro.param('d', torch.ones((1, item_size)) - 0.1,
                                               constraint=constraints.unit_interval)
        with pyro.plate("data", sample_size) as ind:
            irt_param_kwargs['x'] = pyro.sample(
                'x',
                dist.MultivariateNormal(
                    torch.zeros((len(ind), self.x_feature)),
                    scale_tril=torch.eye(self.x_feature).repeat(len(ind), 1, 1))
            )
            irt_fun = self.IRT_FUN[self._model]
            p = irt_fun(**irt_param_kwargs)
            data_ = data[ind]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, 0), data_)
                p = torch.where(data_nan, torch.full_like(p, 0), p)
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data_)

    def fit(self, optim=Adam({'lr': 5e-2}), loss=Trace_ELBO(num_particles=1), max_iter=5000, random_instance=None):
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)
        with trange(max_iter) as t:
            for i in t:
                t.set_description(f'迭代：{i}')
                loss = svi.step(self.data)
                with torch.no_grad():
                    postfix_kwargs = {}
                    if random_instance is not None:
                        b = pyro.param('b')
                        postfix_kwargs['threshold_error'] = '{0}'.format((b - random_instance.b).pow(2).sqrt().mean())
                        if self._model in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
                            a = pyro.param('a')
                            postfix_kwargs['slop_error'] = '{0}'.format((a - random_instance.a).pow(2).sqrt().mean())
                        if self._model in ('irt_3pl', 'irt_4pl'):
                            c = pyro.param('c')
                            postfix_kwargs['guess_error'] = '{0}'.format((c - random_instance.c).pow(2).sqrt().mean())
                        if self._model == 'irt_4pl':
                            d = pyro.param('d')
                            postfix_kwargs['slip_error'] = '{0}'.format((d - random_instance.d).pow(2).sqrt().mean())
                    t.set_postfix(loss=loss, **postfix_kwargs)


class VaeIRT(BaseIRT):
    # 基于变分自编码器的IRT参数估计
    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = NormEncoder(self.item_size, self.x_feature, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            data_ = data[idx]
            data_nan = torch.isnan(data_)
            if data_nan.any():
                data_ = torch.where(data_nan, torch.full_like(data_, -1), data_)
            x_local, x_scale_tril = self.encoder.forward(data_)
            pyro.sample('x', dist.MultivariateNormal(x_local, scale_tril=x_scale_tril))


class VIRT(BaseIRT):
    # 基于黑盒变分推断的IRT参数估计
    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        x_local = pyro.param('x_local', torch.zeros((sample_size, self.x_feature)))
        x_scale_tril = pyro.param(
            'x_scale',
            torch.eye(self.x_feature).repeat(sample_size, 1, 1),
            constraint=constraints.lower_cholesky
        )
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            pyro.sample('x', dist.MultivariateNormal(x_local[idx], scale_tril=x_scale_tril[idx]))


class BaseCDM(BasePsy):

    CDM_FUN = {
        'dina': dina,
        'dino': dino
    }

    def __init__(self, q, model='dina', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.attr_size = q.size(0)
        self._model = model

    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        with pyro.plate("data", sample_size) as ind:
            attr = pyro.sample(
                'attr',
                dist.Bernoulli(torch.zeros((len(ind), self.attr_size)) + 0.5).to_event(1)
            )
            p = self.CDM_FUN[self._model](attr, self.q, g, s)
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[ind])

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
                        g = pyro.param('g')
                        s = pyro.param('s')
                        postfix_kwargs.update({
                            'g': '{0}'.format((g - random_instance.g).abs().mean()),
                            's': '{0}'.format((s - random_instance.s).abs().mean())
                        })
                    t.set_postfix(loss=loss, **postfix_kwargs)


class VaeCDM(BaseCDM):
    # 基于变分自编码器的CDM参数估计
    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = BinEncoder(self.item_size, self.attr_size, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            attr_p = self.encoder.forward(data[idx])
            pyro.sample(
                'attr',
                dist.Bernoulli(attr_p).to_event(1)
            )


class VCDM(BaseCDM):
    # 基于黑盒变分推断的CDM参数估计
    def guide(self, data):
        sample_size = self.sample_size
        attr_p = pyro.param('attr_p', torch.zeros((sample_size, self.attr_size)) + 0.5, constraint=constraints.unit_interval)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            pyro.sample(
                'attr',
                dist.Bernoulli(attr_p[idx]).to_event(1)
            )


class VCCDM(BaseCDM):
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
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[idx])

    def guide(self, data):
        pass

    def fit(self, loss=TraceEnum_ELBO(num_particles=1), *args, **kwargs):
        super().fit(loss=loss, *args, **kwargs)


class VaeCCDM(VCCDM):
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
            attr_p = self.encoder.forward(data[idx])
            attr_idx = pyro.sample(
                'attr_idx',
                dist.Categorical(attr_p).to_event(0)
            )
            p = all_p[attr_idx]
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[idx])


class VCHoDina(VCCDM):
    # 基于离散潜变量黑盒变分推断的HO-DINA参数估计，效果绝佳

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        lam0 = pyro.param('lam0', torch.zeros((1, 5)))
        lam1 = pyro.param('lam1', torch.ones((1, 5)), constraint=constraints.positive)
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        all_p = dina(self.all_attr, self.q, g, s)
        with pyro.plate("data", sample_size) as ind:
            theta = pyro.sample(
                'theta',
                dist.Normal(torch.zeros((len(ind), 1)), torch.ones((len(ind), 1))).to_event(1)
            )
            attr_p = torch.sigmoid(theta.mm(lam1) + lam0)
            likelihood_attr_p = torch.exp(torch.log(attr_p).mm(self.all_attr.T) + torch.log(1 - attr_p).mm(1 - self.all_attr.T))
            attr_idx = pyro.sample(
                'attr_idx',
                dist.Categorical(likelihood_attr_p).to_event(0)
            )
            p = all_p[attr_idx]
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[ind])

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
                            'lam1': '{0}'.format((lam1 - random_instance.lam1).pow(2).sqrt().mean()),
                        })
                    t.set_postfix(loss=loss, **postfix_kwargs)


class VaeCHoDina(VCHoDina):
    # 基于离散潜变量变分自编码器的HO-DINA参数估计，效果绝佳

    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = NormEncoder(self.item_size, 1, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            theta_local, theta_scale = self.encoder.forward(data[idx])
            pyro.sample(
                'theta',
                dist.Normal(theta_local, theta_scale).to_event(1)
            )
