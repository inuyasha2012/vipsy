from math import nan
import random
from unittest import TestCase

import numpy as np
import pyro
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, HMC, NUTS, MCMC
from pyro.optim import Adam, StepLR, MultiStepLR, PyroLRScheduler
import torch
# from sklearn.impute import KNNImputer
from vi import RandomIrt1PL, RandomIrt2PL, RandomIrt3PL, RandomIrt4PL, RandomDina, RandomDino, RandomHoDina, \
    VaeIRT, VIRT, VCDM, VaeCDM, VCCDM, VaeCCDM, VCHoDina, VaeCHoDina
from pyro import distributions as dist


class TestMixin(object):

    def prepare_cuda(self):
        self.cuda = torch.cuda.is_available()
        print('cuda: {0}'.format(torch.cuda.is_available()))
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')


class IRTRandomMixin(object):

    def gen_sample(self, random_class, sample_size, **kwargs):
        i = kwargs.pop('file_index', '')
        random_instance = random_class(sample_size=sample_size, **kwargs)
        y = random_instance.y
        np.savetxt(f'{random_class.name or "data"}_{sample_size}_{i}.txt', y.numpy())
        np.savetxt(f'{random_class.name or "data"}_{sample_size}_a_{i}.txt', random_instance.a.numpy())
        np.savetxt(f'{random_class.name or "data"}_{sample_size}_b_{i}.txt', random_instance.b.numpy())
        np.savetxt(f'{random_class.name or "data"}_{sample_size}_c_{i}.txt', random_instance.c.numpy())
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

    def test_bbvi_iter_10_item_50_sample_100(self):
        for i in range(10):
            y = torch.from_numpy(np.loadtxt(f'irt_2pl_100_{i}.txt')).float()
            b = torch.from_numpy(np.loadtxt(f'irt_2pl_100_b_{i}.txt'))
            a = torch.from_numpy(np.loadtxt(f'irt_2pl_100_a_{i}.txt'))
            r = RandomIrt2PL()
            r.a = a.T
            r.b = b.T
            model = VIRT(data=y, model='irt_2pl')
            model.fit(random_instance=r, optim=Adam({'lr': 1e-2}), max_iter=10000)
            a = pyro.param('a')
            b = pyro.param('b')
            print('a:{0}'.format((a - r.a).pow(2).sqrt().sum() / 50))
            print('b:{0}'.format((b - r.b).pow(2).sqrt().mean()))
            pyro.clear_param_store()

    def test_bbvi_iter_10_item_100_sample_200(self):
        a_rmse_l = []
        b_rmse_l = []
        for i in range(10):
            y = torch.from_numpy(np.loadtxt(f'irt_2pl_200_{i}.txt')).float()
            b = torch.from_numpy(np.loadtxt(f'irt_2pl_200_b_{i}.txt'))
            a = torch.from_numpy(np.loadtxt(f'irt_2pl_200_a_{i}.txt'))
            r = RandomIrt2PL()
            r.a = a.T
            r.b = b.T
            model = VIRT(data=y, model='irt_2pl')
            model.fit(random_instance=r, optim=Adam({'lr': 1e-3}), max_iter=20000)
            a = pyro.param('a')
            b = pyro.param('b')
            a_rmse = (a - r.a).pow(2).sqrt().sum() / y.shape[1]
            a_rmse_l.append(a_rmse.item())
            print('a:{0}'.format(a_rmse))
            b_rmse = (b - r.b).pow(2).sqrt().mean()
            b_rmse_l.append(b_rmse.item())
            print('b:{0}'.format(b_rmse))
            pyro.clear_param_store()
        print(f'a_mean:{np.mean(a_rmse_l)}')
        print(f'a_std:{np.std(a_rmse_l)}')
        print(f'b_mean:{np.mean(b_rmse_l)}')
        print(f'b_std:{np.std(b_rmse_l)}')

    def test_bbvi_iter_10_item_50_sample_500(self):
        a_rmse_l = []
        b_rmse_l = []
        for i in range(10):
            y = torch.from_numpy(np.loadtxt(f'irt_2pl_500_{i}.txt')).float()
            b = torch.from_numpy(np.loadtxt(f'irt_2pl_500_b_{i}.txt'))
            a = torch.from_numpy(np.loadtxt(f'irt_2pl_500_a_{i}.txt'))
            r = RandomIrt2PL()
            r.a = a.T
            r.b = b.T
            model = VIRT(data=y, model='irt_2pl')
            model.fit(random_instance=r, optim=Adam({'lr': 1e-2}), max_iter=10000)
            a = pyro.param('a')
            b = pyro.param('b')
            a_rmse = (a - r.a).pow(2).sqrt().sum() / y.shape[1]
            a_rmse_l.append(a_rmse.item())
            print('a:{0}'.format(a_rmse))
            b_rmse = (b - r.b).pow(2).sqrt().mean()
            b_rmse_l.append(b_rmse.item())
            print('b:{0}'.format(b_rmse))
            pyro.clear_param_store()
        print(f'a_mean:{np.mean(a_rmse_l)}')
        print(f'a_std:{np.std(a_rmse_l)}')
        print(f'b_mean:{np.mean(b_rmse_l)}')
        print(f'b_std:{np.std(b_rmse_l)}')

    def test_ai_iter_10_item_50_sample_100(self):
        for i in range(10):
            item_size = 50
            sample_size = 100
            y, random_instance = self.gen_sample(RandomIrt2PL, sample_size, item_size=item_size, file_index=i)
            model = VaeIRT(data=y, model='irt_2pl', subsample_size=20)
            model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-3}), max_iter=20000)
            a = pyro.param('a')
            b = pyro.param('b')
            print('a:{0}'.format((a - random_instance.a).pow(2).sqrt().sum() / item_size))
            print('b:{0}'.format((b - random_instance.b).pow(2).sqrt().mean()))
            pyro.clear_param_store()

    def test_ai_iter_10_item_100_sample_200(self):
        for i in range(10):
            item_size = 100
            sample_size = 200
            y, random_instance = self.gen_sample(RandomIrt2PL, sample_size, item_size=item_size, file_index=i)
            model = VaeIRT(data=y, model='irt_2pl', subsample_size=20)
            model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-3}), max_iter=20000)
            a = pyro.param('a')
            b = pyro.param('b')
            print('a:{0}'.format((a - random_instance.a).pow(2).sqrt().sum() / item_size))
            print('b:{0}'.format((b - random_instance.b).pow(2).sqrt().mean()))
            pyro.clear_param_store()

    def test_ai_iter_10_item_50_sample_500(self):
        a_rmse_l = []
        b_rmse_l = []
        for i in range(10):
            item_size = 50
            sample_size = 500
            y, random_instance = self.gen_sample(RandomIrt2PL, sample_size, item_size=item_size, file_index=i)
            model = VaeIRT(data=y, model='irt_2pl', subsample_size=100)
            model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-3}), max_iter=10000)
            a = pyro.param('a')
            b = pyro.param('b')
            a_rmse = (a - random_instance.a).pow(2).sqrt().sum() / item_size
            a_rmse_l.append(a_rmse.item())
            b_rmse = (b - random_instance.b).pow(2).sqrt().mean()
            b_rmse_l.append(b_rmse.item())
            print('a:{0}'.format(a_rmse))
            print('b:{0}'.format(b_rmse))
            pyro.clear_param_store()
        print(f'a_mean:{np.mean(a_rmse_l)}')
        print(f'a_std:{np.std(a_rmse_l)}')
        print(f'b_mean:{np.mean(b_rmse_l)}')
        print(f'b_std:{np.std(b_rmse_l)}')


class Irt2PLMissingTestCase(TestCase, TestMixin, IRTRandomMixin):
    # 缺失数据下的变分推断

    def setUp(self):
        self.prepare_cuda()

    def gen_missing_y(self, sample_size=1000, missing_rate=0.1, **kwargs):
        row_y, random_instance = self.gen_sample(RandomIrt2PL, sample_size, **kwargs)
        y_size = row_y.size(0) * row_y.size(1)
        row_idx = torch.arange(0, row_y.size(0)).repeat(int(row_y.size(1) * missing_rate))
        col_idx = torch.randint(0, row_y.size(1), (int(y_size * missing_rate),))
        row_y[row_idx, col_idx] = nan
        return row_y, random_instance

    def test_bbvi(self):
        y, random_instance = self.gen_missing_y(sample_size=10000, missing_rate=0.1, item_size=100)
        model = VIRT(data=y, model='irt_2pl', subsample_size=10000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        random_class = RandomIrt2PL
        sample_size = 1000
        y, random_instance = self.gen_missing_y(sample_size=sample_size, missing_rate=0.9, item_size=1000)
        np.savetxt(f'{random_class.name or "data"}_{sample_size}.txt', y.numpy())
        np.savetxt(f'{random_class.name or "data"}_{sample_size}_a.txt', random_instance.a.numpy())
        np.savetxt(f'{random_class.name or "data"}_{sample_size}_b.txt', random_instance.b.numpy())
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam(self.optim), max_iter=100000)

    @staticmethod
    def optim(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-2}
        if param_name == 'b':
            return {'lr': 1e-3}
        return {'lr': 1e-3}


class IrtMultiDimTestCase(TestCase, TestMixin, IRTRandomMixin):
    # 多维项目反应理论

    def setUp(self):
        self.prepare_cuda()

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
                a[a < 0.01] = 0.01
            else:
                x_size = item_size - j
                sigma = self.gen_omega(x_size)
                a[:x_size, j] = mdisc[j] * torch.cos(torch.FloatTensor(sigma))
                a[a < 0.01] = 0.01
        for i in range(x_feature):
            a[i, item_size - i:] = 0
        return a

    def test_ai_100_dim_2pl(self):
        sample_size = 10000
        subsample_size = 100
        item_size = 500
        x_feature = 100
        random_instance = RandomIrt2PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        mdisc = torch.FloatTensor(item_size).log_normal_(0, 0.5)
        mdiff = torch.FloatTensor(item_size).normal_(0.5, 1)
        a = self.gen_a(item_size, mdisc, x_feature)
        random_instance.a = a
        random_instance.b = -mdiff * mdisc
        random_instance.b = random_instance.b.view(1, -1)
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=subsample_size, x_feature=x_feature, D=1.702)
        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': self.optim_2PL,
                                 'milestones': [
                                     # int(sample_size / subsample_size) * 50,
                                     int(sample_size / subsample_size) * 190
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 200), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    @staticmethod
    def optim_2PL(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-2}
        if param_name == 'b':
            return {'lr': 1e-2}
        return {'lr': 1e-3}

    def test_ai_5_dim_3pl(self):
        sample_size = 10000
        subsample_size = 20
        item_size = 50
        x_feature = 10
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
                                 'optim_args': self.optim_3PL,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 190,
                                     # int(sample_size / subsample_size) * 400
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 200), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    @staticmethod
    def optim_3PL(module_name, param_name):
        if param_name == 'a':
            return {'lr': 1e-2}
        if param_name == 'b':
            return {'lr': 1e-2}
        if param_name == 'c':
            return {'lr': 1e-2}
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
        model.fit(optim=Adam({'lr': 1e-2}), max_iter=10000, loss=Trace_ELBO(num_particles=20))

    def test_ai_5_dim(self):
        sample_size = 10000
        subsample_size = 100
        item_size = 50
        x_feature = 2
        random_instance = RandomIrt2PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        for i in range(x_feature):
            random_instance.a[i, item_size - i:] = 0
        # y = random_instance.y.cuda()
        # random_instance.a = random_instance.a.cuda()
        # random_instance.b = random_instance.b.cuda()
        scale_tril = torch.eye(x_feature)
        scale_tril[0, 1] = scale_tril[1, 0] = 0.7
        random_instance.x = dist.MultivariateNormal(
                            torch.zeros((sample_size, x_feature)),
                            scale_tril=scale_tril
                        ).sample()
        y = random_instance.y
        np.savetxt(f'irt_2pl_{sample_size}.txt', y.numpy())
        np.savetxt(f'irt_2pl_{sample_size}_a.txt', random_instance.a.T.numpy())
        np.savetxt(f'irt_2pl_{sample_size}_b.txt', random_instance.b.T.numpy())
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=subsample_size, x_feature=x_feature,
                       hidden_dim=64)

        def optim(_, param_name):
            if param_name == 'a':
                return {'lr': 1e-4}
            if param_name == 'b':
                return {'lr': 1e-3}
            return {'lr': 1e-4}

        model.fit(optim=Adam(optim), max_iter=int(sample_size / subsample_size * 100000), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))


class Irt3PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt3PL, 10000)
        model = VIRT(data=y, model='irt_3pl')
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=50000)

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt3PL, 100)
        model = VaeIRT(data=y, model='irt_3pl', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-4}), max_iter=50000)

    def test_ai_iter_10_item_50_sample_500(self):
        a_rmse_l = []
        b_rmse_l = []
        c_rmse_l = []
        for i in range(10):
            item_size = 50
            sample_size = 500
            y, random_instance = self.gen_sample(RandomIrt3PL, sample_size, item_size=item_size, file_index=i)
            model = VaeIRT(data=y, model='irt_3pl', subsample_size=100)

            def optim(_, pn):
                if pn in ('a', 'b'):
                    return {'lr': 1e-3}
                return {'lr': 1e-3}

            model.fit(random_instance=random_instance, optim=Adam(optim), max_iter=10000)
            a = pyro.param('a')
            b = pyro.param('b')
            c = pyro.param('c')
            a_rmse = (a - random_instance.a).pow(2).sqrt().sum() / item_size
            a_rmse_l.append(a_rmse.item())
            b_rmse = (b - random_instance.b).pow(2).sqrt().mean()
            b_rmse_l.append(b_rmse.item())
            c_rmse = (c - random_instance.c).pow(2).sqrt().mean()
            c_rmse_l.append(c_rmse.item())
            print('a:{0}'.format(a_rmse))
            print('b:{0}'.format(b_rmse))
            print('c:{0}'.format(c_rmse))
            pyro.clear_param_store()
        print(f'a_mean:{np.mean(a_rmse_l)}')
        print(f'a_std:{np.std(a_rmse_l)}')
        print(f'b_mean:{np.mean(b_rmse_l)}')
        print(f'b_std:{np.std(b_rmse_l)}')
        print(f'c_mean:{np.mean(c_rmse_l)}')
        print(f'c_std:{np.std(c_rmse_l)}')


class Irt4PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt4PL, 10000)
        model = VIRT(data=y, model='irt_4pl', subsample_size=10000)
        model.fit(random_instance=random_instance, max_iter=50000, optim=Adam({'lr': 1e-3}))

    def test_ai(self):
        sample_size = 100000
        y, random_instance = self.gen_sample(RandomIrt4PL, sample_size)
        subsample_size = 100
        model = VaeIRT(data=y, model='irt_4pl', subsample_size=subsample_size)
        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': {'lr': 1e-2},
                                 'milestones': [
                                     int(sample_size / subsample_size) * 40,
                                     int(sample_size / subsample_size) * 70
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(random_instance=random_instance, optim=scheduler, max_iter=int(sample_size / subsample_size) * 100)


class CDMRandomMixin(object):

    def gen_sample(self, random_class, sample_size, **kwargs):
        random_instance = random_class(sample_size=sample_size, **kwargs)
        y = random_instance.y
        q = random_instance.q
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}.txt', y.numpy())
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}_q.txt', q.numpy())
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}_g.txt', random_instance.g.numpy())
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}_s.txt', random_instance.s.numpy())
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}_lam0.txt', random_instance.lam0.numpy())
        # np.savetxt(f'{random_class.name or "data"}_{sample_size}_lam1.txt', random_instance.lam1.numpy())
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
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-3}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomDina, 10000, q_size=20, item_size=2000)
        model = VaeCDM(data=y, q=q, model='dina', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam(self.optim_ai), max_iter=100000)

    @staticmethod
    def optim_ai(_, param_name):
        if param_name in ('g'):
            return {"lr": 1e-3}
        return {'lr': 5e-4}


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
        y, q, random_instance = self.gen_sample(RandomDina, 100000, q_size=10)
        model = VaeCCDM(data=y, q=q, model='dina', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))


class PaDinaMissingTestCase(TestCase, TestMixin, CDMRandomMixin):
    # 缺失数据下的变分推断

    def setUp(self):
        self.prepare_cuda()

    def gen_missing_y(self, sample_size=1000, missing_rate=0.1, **kwargs):
        row_y, q, random_instance = self.gen_sample(RandomHoDina, sample_size, **kwargs)
        y_size = row_y.size(0) * row_y.size(1)
        row_idx = torch.arange(0, row_y.size(0)).repeat(int(row_y.size(1) * missing_rate))
        col_idx = torch.randint(0, row_y.size(1), (int(y_size * missing_rate),))
        row_y[row_idx, col_idx] = nan
        return row_y, q, random_instance

    def test_bbvi(self):
        y, q, random_instance = self.gen_missing_y(sample_size=10000, missing_rate=0.9, item_size=1000)
        model = VCCDM(data=y, q=q, model='dina', subsample_size=100)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))

    def test_ai(self):
        sample_size = 1000
        y, q, random_instance = self.gen_missing_y(sample_size=sample_size, missing_rate=0.9, item_size=500)
        # np.savetxt(f'ho_dina_{sample_size}.txt', y.numpy())
        # np.savetxt(f'ho_dina_{sample_size}_q.txt', q.numpy())
        # np.savetxt(f'ho_dina_{sample_size}_g.txt', random_instance.g.numpy())
        # np.savetxt(f'ho_dina_{sample_size}_s.txt', random_instance.s.numpy())
        # np.savetxt(f'ho_dina_{sample_size}_lam0.txt', random_instance.lam0.numpy())
        # np.savetxt(f'ho_dina_{sample_size}_lam1.txt', random_instance.lam1.numpy())
        subsample_size = 20
        model = VaeCHoDina(data=y, q=q, model='dina', subsample_size=subsample_size)
        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': self.optim,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 50,
                                     int(sample_size / subsample_size) * 70
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(random_instance=random_instance, optim=scheduler, loss=TraceEnum_ELBO(num_particles=10))

    @staticmethod
    def optim(_, p_n):
        if p_n in ('g', 's'):
            return {"lr": 1e-1}
        if p_n in ('lam0', 'lam1'):
            return {'lr': 1e-1}
        return {'lr': 1e-3}

class PaDinoTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomDino, 1000)
        model = VCCDM(data=y, q=q, model='dino', subsample_size=1000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))

    def test_ai(self):
        y, q, random_instance = self.gen_sample(RandomDino, 10000)
        model = VaeCCDM(data=y, q=q, model='dino', subsample_size=50)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-3}))


class PaHoDinaTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, q, random_instance = self.gen_sample(RandomHoDina, 1000)
        model = VCHoDina(data=y, q=q, subsample_size=1000)
        model.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        sample_size = 10000
        y, q, random_instance = self.gen_sample(RandomHoDina, sample_size)
        subsample_size = 100
        model = VaeCHoDina(data=y, q=q, subsample_size=subsample_size)

        def optim(_, pn):
            if pn in ('lam1', 'lam0', 'g', 's'):
                return {'lr': 1e-1}
            return {'lr': 1e-3}

        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': optim,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 40,
                                     int(sample_size / subsample_size) * 50,
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(random_instance=random_instance, optim=scheduler)