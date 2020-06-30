import torch
import torch.distributions
import pyro
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.infer import SVI, Trace_ELBO, config_enumerate, TraceEnum_ELBO
from pyro.optim import Adam
from torch import nn
from tqdm import trange

# ======心理测量模型 start=============


def irt_1pl(x, b):
    return torch.sigmoid(x + b)


def irt_2pl(x, a, b):
    return torch.sigmoid(x.mm(a) + b)


def irt_3pl(x, a, b, c):
    return c + (1 - c) * irt_2pl(x, a, b)


def irt_4pl(x, a, b, c, d):
    return c + (d - c) * irt_2pl(x, a, b)


def dina(skill, attr, g, s):
    yita = skill.mm(attr)
    aa = (attr ** 2).sum(dim=0)
    yita[yita < aa] = 0
    yita[yita == aa] = 1
    p = (1 - s) ** yita * g ** (1 - yita)
    return p


def dino(skill, attr, g, s):
    yita = (1 - skill).mm(attr)
    aa = (attr ** 2).sum(dim=0)
    yita[yita < aa] = 1
    yita[yita == aa] = 0
    p = (1 - s) ** yita * g ** (1 - yita)
    return p


def ho_dina(lam0, lam1, theta, attr, g, s):
    skill_p = torch.sigmoid(theta.mm(lam1) + lam0)
    skill = dist.Bernoulli(skill_p).sample()
    return dina(skill, attr, g, s)

# ======心理测量模型 end=============

# ======随机数据生成 start=======


class RandomPsyData(object):

    def __init__(self, sample_size=10000, item_size=100):
        self.item_size = item_size
        self.sample_size = sample_size


class RandomDina(RandomPsyData):

    name = 'dina'

    def __init__(self, attr_size=5, attr_p=0.5, skill_p=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_size = attr_size
        self.attr = torch.FloatTensor(attr_size, self.item_size).bernoulli_(attr_p)
        self.skill = torch.FloatTensor(self.sample_size, attr_size).bernoulli_(skill_p)
        self.g = torch.FloatTensor(1, self.item_size).uniform_(0, 0.3)
        self.s = torch.FloatTensor(1, self.item_size).uniform_(0, 0.3)

    @property
    def y(self):
        p = dina(self.skill, self.attr, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomDino(RandomDina):

    name = 'dino'

    @property
    def y(self):
        p = dino(self.skill, self.attr, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomHoDina(RandomDina):

    name = 'ho_dina'

    def __init__(self, theta_dim=1, theta_local=0, theta_scale=1, lam0_local=0, lam0_scale=1, lam1_lower=0.5,
                 lam1_upper=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = torch.FloatTensor(self.sample_size, theta_dim).normal_(theta_local, theta_scale)
        self.lam0 = torch.FloatTensor(theta_dim, self.attr_size).normal_(lam0_local, lam0_scale)
        self.lam1 = torch.FloatTensor(theta_dim, self.attr_size).uniform_(lam1_lower, lam1_upper)

    @property
    def y(self):
        p = ho_dina(self.lam0, self.lam1, self.theta, self.attr, self.g, self.s)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt1PL(RandomPsyData):

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
        super().__init__(*args, **kwargs)
        self.x_feature = x_feature
        self.x = torch.FloatTensor(self.sample_size, x_feature).normal_(x_local, x_scale)
        self.b = torch.FloatTensor(1, self.item_size).normal_(b_local, b_scale)

    @property
    def y(self):
        p = irt_1pl(self.x, self.b)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt2PL(RandomIrt1PL):

    name = 'irt_2pl'

    def __init__(
            self,
            a_lower=1,
            a_upper=3,
            *args,
            **kwargs
    ):
        super(RandomIrt2PL, self).__init__(*args, **kwargs)
        self.a = torch.FloatTensor(self.x_feature, self.item_size).uniform_(a_lower, a_upper)

    @property
    def y(self):
        p = irt_2pl(self.x, self.a, self.b)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt3PL(RandomIrt2PL):

    name = 'irt_3pl'

    def __init__(self, c_unif_lower=0.05, c_unif_upper=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = torch.FloatTensor(1, self.item_size).uniform_(c_unif_lower, c_unif_upper)

    @property
    def y(self):
        p = irt_3pl(self.x, self.a, self.b, self.c)
        return torch.FloatTensor(*p.size()).bernoulli_(p)


class RandomIrt4PL(RandomIrt3PL):

    name = 'irt_4pl'

    def __init__(self, d_unif_lower=0.8, d_unif_upper=0.95, *args, **kwargs):
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

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        x_loc = self.fc21(hidden)
        x_scale = torch.exp(self.fc22(hidden))
        return x_loc, x_scale


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


class BasePsy(nn.Module):

    def __init__(self, data, subsample_size=None):
        super().__init__()
        self.data = data
        self.sample_size = data.size(0)
        self.item_size = data.size(1)
        self.subsample_size = subsample_size if subsample_size is not None else self.sample_size


class BaseIRT(BasePsy):

    IRT_FUN = {
        'irt_1pl': irt_1pl,
        'irt_2pl': irt_2pl,
        'irt_3pl': irt_3pl,
        'irt_4pl': irt_4pl,
    }

    def __init__(self, model='irt_2pl', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = model

    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        irt_param_kwargs = {'b': pyro.param('b', torch.zeros((1, item_size)))}
        if self._model in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
            irt_param_kwargs['a'] = pyro.param('a', torch.ones((1, item_size)))
        if self._model in ('irt_3pl', 'irt_4pl'):
            irt_param_kwargs['c'] = pyro.param('c', torch.zeros((1, item_size)))
        if self._model == 'irt_4pl':
            irt_param_kwargs['d'] = pyro.param('d', torch.ones((1, item_size)))
        with pyro.plate("data", sample_size) as ind:
            irt_param_kwargs['x'] = pyro.sample(
                'x',
                dist.Normal(torch.zeros((len(ind), 1)), torch.ones((len(ind), 1))).to_event(1)
            )
            irt_fun = self.IRT_FUN[self._model]
            p = irt_fun(**irt_param_kwargs)
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[ind])

    def fit(self, optim=Adam({'lr': 5e-2}), loss=Trace_ELBO(num_particles=1), max_iter=5000, random_instance=None):
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)
        with trange(max_iter) as t:
            for i in t:
                t.set_description(f'迭代：{i}')
                svi.step(self.data)
                loss = svi.evaluate_loss(self.data)
                with torch.no_grad():
                    postfix_kwargs = {}
                    if random_instance is not None:
                        b = pyro.param('b')
                        postfix_kwargs['threshold_error'] = '{0}'.format((b - random_instance.b).abs().mean())
                        if self._model in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
                            a = pyro.param('a')
                            postfix_kwargs['slop_error'] = '{0}'.format((a - random_instance.a).abs().mean())
                        if self._model in ('irt_3pl', 'irt_4pl'):
                            c = pyro.param('c')
                            postfix_kwargs['guess_error'] = '{0}'.format((c - random_instance.c).abs().mean())
                        if self._model == 'irt_4pl':
                            d = pyro.param('d')
                            postfix_kwargs['slip_error'] = '{0}'.format((d - random_instance.d).abs().mean())
                    t.set_postfix(loss=loss, **postfix_kwargs)


class VaeIRT(BaseIRT):

    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = NormEncoder(self.item_size, 1, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            x_local, x_scale = self.encoder.forward(data[idx])
            pyro.sample('x', dist.Normal(x_local, x_scale).to_event(1))


class VIRT(BaseIRT):

    def guide(self, data):
        sample_size = self.sample_size
        subsample_size = self.subsample_size
        x_local = pyro.param('x_local', torch.zeros((sample_size, 1)))
        x_scale = pyro.param('x_scale', torch.ones((sample_size, 1)), constraint=constraints.positive)
        with pyro.plate("data", sample_size, subsample_size=subsample_size) as idx:
            pyro.sample('x', dist.Normal(x_local[idx], x_scale[idx]).to_event(1))


class BaseCDM(BasePsy):

    CDM_FUN = {
        'dina': dina,
        'dino': dino
    }

    def __init__(self, attr, model='dina', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr = attr
        self.skill_size = attr.size(0)
        self._model = model

    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        with pyro.plate("data", sample_size) as ind:
            skill = pyro.sample(
                'skill',
                dist.Bernoulli(torch.zeros((len(ind), self.skill_size)) + 0.5).to_event(1)
            )
            p = self.CDM_FUN[self._model](skill, self.attr, g, s)
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
    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = BinEncoder(self.item_size, self.skill_size, hidden_dim)

    def guide(self, data):
        sample_size = self.sample_size
        pyro.module('encoder', self.encoder)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            skill_p = self.encoder.forward(data[idx])
            pyro.sample(
                'skill',
                dist.Bernoulli(skill_p).to_event(1)
            )


class VCDM(BaseCDM):

    def guide(self, data):
        sample_size = self.sample_size
        skill_p = pyro.param('skill_p', torch.zeros((sample_size, self.skill_size)) + 0.5, constraint=constraints.unit_interval)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            pyro.sample(
                'skill',
                dist.Bernoulli(skill_p[idx]).to_event(1)
            )


class VCCDM(BaseCDM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_skill = self.get_all_skills()

    def get_all_skills(self):
        row_size = 2 ** self.skill_size
        all_skill = torch.zeros((row_size, self.skill_size))
        for i in range(row_size):
            num = i
            count = 0
            while True:
                if num == 0:
                    break
                num, rem = divmod(num, 2)
                all_skill[i][count] = rem
                count += 1
        return all_skill

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        all_p = self.CDM_FUN[self._model](self.all_skill, self.attr, g, s)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            skill_idx = pyro.sample(
                'skill_idx',
                dist.Categorical(torch.zeros((len(idx), self.all_skill.size(0))) + 1 / self.all_skill.size(0)).to_event(0)
            )
            p = all_p[skill_idx]
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[idx])

    def guide(self, data):
        pass

    def fit(self, loss=TraceEnum_ELBO(num_particles=1), *args, **kwargs):
        super().fit(loss=loss, *args, **kwargs)


class VaeCCDM(VCCDM):

    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = SoftmaxEncoder(self.item_size, self.all_skill.size(0), hidden_dim)

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        pyro.module('encoder', self.encoder)
        all_p = self.CDM_FUN[self._model](self.all_skill, self.attr, g, s)
        with pyro.plate("data", sample_size, subsample_size=self.subsample_size) as idx:
            skill_p = self.encoder.forward(data[idx])
            skill_idx = pyro.sample(
                'skill_idx',
                dist.Categorical(skill_p).to_event(0)
            )
            p = all_p[skill_idx]
            pyro.sample('y', dist.Bernoulli(p).to_event(1), obs=data[idx])


class VCHoDina(VCCDM):

    def __init__(self, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = NormEncoder(self.item_size, 1, hidden_dim)

    @config_enumerate
    def model(self, data):
        item_size = self.item_size
        sample_size = self.sample_size
        lam0 = pyro.param('lam0', torch.zeros((1, 5)))
        lam1 = pyro.param('lam1', torch.ones((1, 5)), constraint=constraints.positive)
        g = pyro.param('g', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        s = pyro.param('s', torch.zeros((1, item_size)) + 0.1, constraint=constraints.interval(0, 1))
        all_p = dina(self.all_skill, self.attr, g, s)
        with pyro.plate("data", sample_size) as ind:
            theta = pyro.sample(
                'theta',
                dist.Normal(torch.zeros((len(ind), 1)), torch.ones((len(ind), 1))).to_event(1)
            )
            skill_p = torch.sigmoid(theta.mm(lam1) + lam0)
            likelihood_skill_p = torch.exp(torch.log(skill_p).mm(self.all_skill.T) + torch.log(1 - skill_p).mm(1 - self.all_skill.T))
            skill_idx = pyro.sample(
                'skill_idx',
                dist.Categorical(likelihood_skill_p).to_event(0)
            )
            p = all_p[skill_idx]
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
