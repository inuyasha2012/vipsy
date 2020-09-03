from math import nan
import random
from unittest import TestCase

import numpy as np
from pyro.infer import Trace_ELBO
from pyro.optim import Adam, StepLR, ReduceLROnPlateau, MultiStepLR, PyroLRScheduler
import torch
# from sklearn.impute import KNNImputer
from vi import RandomIrt1PL, RandomIrt2PL, RandomIrt3PL, RandomIrt4PL, RandomDina, RandomDino, RandomHoDina, \
    VaeIRT, VIRT, VCDM, VaeCDM, VCCDM, VaeCCDM, VCHoDina, VaeCHoDina


class TestMixin(object):

    def prepare_cuda(self):
        self.cuda = torch.cuda.is_available()
        print('cuda: {0}'.format(torch.cuda.is_available()))
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')


class IRTRandomMixin(object):

    def gen_sample(self, random_class, sample_size, **kwargs):
        random_instance = random_class(sample_size=sample_size, **kwargs)
        y = random_instance.y
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}.txt', y.numpy())
        if self.cuda:
            y = y.cuda()
            random_instance.a = random_instance.a.cuda()
            random_instance.b = random_instance.b.cuda()
        return y, random_instance


class Irt1PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt1PL, 1000)
        model = VIRT(data=y, model='irt_1pl', subsample_size=1000)
        model.fit(random_instance=random_instance)

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt1PL, 100000)
        model = VaeIRT(data=y, model='irt_1pl', subsample_size=100)
        model.fit(random_instance=random_instance)


class Irt2PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt2PL, 10000)
        model = VIRT(data=y, model='irt_2pl')
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=50000)

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt2PL, 100000)
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=50000)


class Irt2PLMissingTestCase(TestCase, TestMixin, IRTRandomMixin):
    # 缺失数据下的变分推断

    def setUp(self):
        self.prepare_cuda()

    def gen_missing_y(self, sample_size=1000, missing_rate=0.1, **kwargs):
        row_y, random_instance = self.gen_sample(RandomIrt2PL, sample_size, **kwargs)
        y_size = row_y.size(0) * row_y.size(1)
        # row_idx = torch.randint(0, row_y.size(0), (int(y_size * missing_rate),))
        row_idx = torch.arange(0, row_y.size(0)).repeat(int(row_y.size(1) * missing_rate))
        col_idx = torch.randint(0, row_y.size(1), (int(y_size * missing_rate),))
        row_y[row_idx, col_idx] = nan
        # imputer = KNNImputer(n_neighbors=5, weights="uniform")
        # y_ = imputer.fit_transform(row_y.numpy())
        # return torch.from_numpy(y_), random_instance
        return row_y, random_instance

    def test_bbvi(self):
        y, random_instance = self.gen_missing_y(sample_size=10000, missing_rate=0.1, item_size=100)
        model = VIRT(data=y, model='irt_2pl', subsample_size=10000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, random_instance = self.gen_missing_y(sample_size=10000, missing_rate=0.9, item_size=1000)
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam(self.optim), max_iter=100000)

    @staticmethod
    def optim(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-2}
        if param_name == 'b':
            return {'lr': 2e-3}
        return {'lr': 2e-3}


class IrtMultiDimTestCase(TestCase, TestMixin, IRTRandomMixin):
    # 多维项目反应理论

    def setUp(self):
        self.prepare_cuda()

    @staticmethod
    def generate_randval(x_low, x_up, x_sum, y):
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
            IrtMultiDimTestCase.generate_randval(x_low, x_up, x_sum, y)

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
            else:
                x_size = item_size - j
                sigma = self.gen_omega(x_size)
                a[:x_size, j] = mdisc[j] * torch.cos(torch.FloatTensor(sigma))
        return a

    def test_ai_100_dim_2pl(self):
        sample_size = 10000
        subsample_size = 100
        item_size = 500
        x_feature = 100
        random_instance = RandomIrt2PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        mdisc = torch.FloatTensor(item_size).log_normal_(0, 0.5)
        mdiff = torch.FloatTensor(item_size).normal_(0, 1)
        a = self.gen_a(item_size, mdisc, x_feature)
        random_instance.a = a
        random_instance.b = -mdiff * mdisc
        random_instance.b = random_instance.b.view(1, -1)
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=subsample_size, x_feature=x_feature)
        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': self.optim_2PL_100_dim,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 50,
                                     int(sample_size / subsample_size) * 60
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 70), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    @staticmethod
    def optim_2PL(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-1}
        if param_name == 'b':
            return {'lr': 1e-1}
        return {'lr': 1e-3}

    def test_ai_5_dim_3pl(self):
        sample_size = 7000
        subsample_size = 100
        item_size = 20
        x_feature = 5
        random_instance = RandomIrt3PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        mdisc = torch.FloatTensor(item_size).log_normal_(0, 0.5)
        mdiff = torch.FloatTensor(item_size).normal_(0.5, 1)
        logit_c = torch.FloatTensor(1, item_size).normal_(-1.39, 0.16)
        c = 1 / (1 + torch.exp(-logit_c))
        a = self.gen_a(item_size, mdisc, x_feature)
        random_instance.a = a
        random_instance.b = -mdiff * mdisc
        random_instance.b = random_instance.b.view(1, -1)
        random_instance.c = c
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_3pl', subsample_size=subsample_size, x_feature=x_feature)
        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': self.optim_3PL_10_dim,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 50,
                                     # int(sample_size / subsample_size) * 400
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 200), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    @staticmethod
    def optim_3PL(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-1}
        if param_name == 'b':
            return {'lr': 1e-1}
        return {'lr': 1e-3}

    def test_ai_10_dim_4pl(self):
        sample_size = 10000
        subsample_size = 100
        item_size = 20
        x_feature = 5
        random_instance = RandomIrt4PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        mdisc = torch.FloatTensor(item_size).log_normal_(0, 0.5)
        mdiff = torch.FloatTensor(item_size).normal_(0.5, 1)
        logit_c = torch.FloatTensor(1, item_size).normal_(-1.39, 0.16)
        logit_d = torch.FloatTensor(1, item_size).normal_(-1.39, 0.16)
        c = 1 / (1 + torch.exp(-logit_c))
        d = 1 / (1 + torch.exp(logit_d))
        a = self.gen_a(item_size, mdisc, x_feature)
        random_instance.a = a
        random_instance.b = -mdiff * mdisc
        random_instance.b = random_instance.b.view(1, -1)
        random_instance.c = c
        random_instance.d = d
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_4pl', subsample_size=subsample_size, x_feature=x_feature)
        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': self.optim_4PL,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 400,
                                     # int(sample_size / subsample_size) * 400
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 500), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    @staticmethod
    def optim_4PL(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-2}
        if param_name == 'b':
            return {'lr': 1e-2}
        return {'lr': 1e-3}

    def test_cfa(self):
        data = np.loadtxt('ex5.2.dat')
        a_free = torch.FloatTensor([
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ])
        model = VIRT(data=torch.from_numpy(data), model='irt_2pl', subsample_size=100, x_feature=2,
                              a_free=a_free.T, a0=a_free.T)
        model.fit(optim=Adam({'lr': 1e-2}), max_iter=10000)


class Irt3PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt3PL, 10000)
        model = VIRT(data=y, model='irt_3pl')
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=50000)

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt3PL, 1000000)
        model = VaeIRT(data=y, model='irt_3pl', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}), max_iter=50000)


class Irt4PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt4PL, 10000)
        model = VIRT(data=y, model='irt_4pl', subsample_size=10000)
        model.fit(random_instance=random_instance, max_iter=50000, optim=Adam({'lr': 1e-2}))

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt4PL, 100000)
        model = VaeIRT(data=y, model='irt_4pl', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=100000)


class CDMRandomMixin(object):

    def gen_sample(self, random_class, sample_size, **kwargs):
        random_instance = random_class(sample_size=sample_size, **kwargs)
        y = random_instance.y
        q = random_instance.q
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}.txt', y.numpy())
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}_q.txt', q.numpy())
        if self.cuda:
            y = y.cuda()
            q = q.cuda()
        return y, q, random_instance


class DinaTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomDina, 1000)
        model = VCDM(data=y, q=q, model='dina', subsample_size=1000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomDina, 100000)
        model = VaeCDM(data=y, q=q, model='dina', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 5e-2}))


class DinoTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomDino, 1000)
        model = VCDM(data=y, q=q, model='dino', subsample_size=1000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomDino, 100000)
        model = VaeCDM(data=y, q=q, model='dino', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=10000)


class PaDinaTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomDina, 1500)
        model = VCCDM(data=y, q=q, model='dina', subsample_size=1500)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomDina, 100000)
        model = VaeCCDM(data=y, q=q, model='dina', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))


class PaDinoTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomDino, 1000)
        model = VCCDM(data=y, q=q, model='dino', subsample_size=1000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomDino, 100000)
        model = VaeCCDM(data=y, q=q, model='dino', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))


class PaHoDinaTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomHoDina, 1000)
        model = VCHoDina(data=y, q=q, subsample_size=1000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomHoDina, 100000)
        model = VaeCHoDina(data=y, q=q, subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-3}))