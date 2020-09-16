import os
from collections import namedtuple
from math import nan
import random
from unittest import TestCase

import numpy as np
import pyro
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, HMC, NUTS, MCMC
from pyro.optim import Adam, StepLR, MultiStepLR, PyroLRScheduler
import torch
from vi import RandomIrt1PL, RandomIrt2PL, RandomIrt3PL, RandomIrt4PL, RandomDina, RandomDino, RandomHoDina, \
    VaeIRT, VIRT, VCDM, VaeCDM, VCCDM, VaeCCDM, VCHoDina, VaeCHoDina, RandomMilIrt2PL, RandomMilIrt3PL, RandomMilIrt4PL
from pyro import distributions as dist
from multiprocessing import Pool


def article_test_load_data_util(
        model_name,
        sample_size,
        item_size,
        x_feature_size,
        file_postfix=0,
        vi_class=VaeIRT,
        vi_class_kwargs=None,
        vi_fit_kwargs=None,
):
    file_prefix = f'irt_{model_name}_sample_{sample_size}_item_{item_size}_dim_{x_feature_size}'
    attrs = ['b']
    if model_name in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
        attrs.append('a')
    if model_name in ('irt_3pl', 'irt_4pl'):
        attrs.append('c')
    if model_name == 'irt_4pl':
        attrs.append('d')
    R = namedtuple('R', attrs)
    y = torch.from_numpy(np.loadtxt(f'{file_prefix}_{file_postfix}.txt')).float()
    r_kwargs = {}
    b = torch.from_numpy(np.loadtxt(f'{file_prefix}_b_{file_postfix}.txt'))
    r_kwargs['b'] = b.T
    if model_name in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
        a = torch.from_numpy(np.loadtxt(f'{file_prefix}_a_{file_postfix}.txt'))
        r_kwargs['a'] = a.T
    if model_name in ('irt_3pl', 'irt_4pl'):
        c = torch.from_numpy(np.loadtxt(f'{file_prefix}_c_{file_postfix}.txt'))
        r_kwargs['c'] = c.T
    if model_name == 'irt_4pl':
        d = torch.from_numpy(np.loadtxt(f'{file_prefix}_d_{file_postfix}.txt'))
        r_kwargs['d'] = d.T
    r = R(**r_kwargs)
    vi_class_kwargs_ = {'data': y, 'model': model_name, 'x_feature': x_feature_size}
    if vi_class_kwargs is not None:
        vi_class_kwargs_.update(vi_class_kwargs)
    vi_model = vi_class(**vi_class_kwargs_)
    vi_fit_kwargs_ = {'optim': Adam({'lr': 1e-2}), 'max_iter': 10000}
    if vi_fit_kwargs is not None:
        vi_fit_kwargs_.update(vi_fit_kwargs)
    vi_model.fit(random_instance=r, **vi_fit_kwargs_)
    rmse_dt = rmse_(item_size, model_name, r, x_feature_size)
    pyro.clear_param_store()
    return rmse_dt


def rmse_(item_size, model_name, r, x_feature):
    rmse_dt = {}
    if model_name in ('irt_2pl', 'irt_3pl', 'irt_4pl'):
        a = pyro.param('a')
        a_rmse = (a - r.a).pow(2).sqrt().sum() / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
        rmse_dt['a'] = a_rmse.item()
        print('a:{0}'.format(a_rmse))
    b = pyro.param('b')
    b_rmse = (b - r.b).pow(2).sqrt().mean()
    rmse_dt['b'] = b_rmse.item()
    print('b:{0}'.format(b_rmse))
    if model_name in ('irt_3pl', 'irt_4pl'):
        c = pyro.param('c')
        c_rmse = (c - r.c).pow(2).sqrt().mean()
        rmse_dt['c'] = c_rmse.item()
        print('c:{0}'.format(c_rmse))
    if model_name == 'irt_4pl':
        d = pyro.param('d')
        d_rmse = (d - r.d).pow(2).sqrt().mean()
        print('d:{0}'.format(d_rmse))
        rmse_dt['d'] = d_rmse.item()
    return rmse_dt


def multiprocess_article_test_load_data_util(
        model_name,
        sample_size,
        item_size,
        x_feature_size,
        vi_class,
        try_count=10,
        vi_class_kwargs=None,
        vi_fit_kwargs=None,
        process_size=2,
        start_idx=0
):
    pool = Pool(processes=process_size)
    res_lt = []
    for i in range(start_idx, start_idx + try_count):
        kwargs = {
            'model_name': model_name,
            'sample_size': sample_size,
            'item_size': item_size,
            'x_feature_size': x_feature_size,
            'vi_class': vi_class,
            'vi_class_kwargs': vi_class_kwargs,
            'vi_fit_kwargs': vi_fit_kwargs,
            'file_postfix': i,
        }
        res = pool.apply_async(func=article_test_load_data_util, kwds=kwargs)
        res_lt.append(res)

    pool.close()
    pool.join()

    print_rmse(res_lt)


def print_rmse(res_lt):
    rmse_dt = {}
    for res in res_lt:
        rmse_dt_ = res.get()
        for key, val in rmse_dt_.items():
            if key in rmse_dt:
                rmse_dt[key].append(val)
            else:
                rmse_dt[key] = [val]
    for k, v in rmse_dt.items():
        print(f'{k}_mean:{np.mean(v)}')
        print(f'{k}_std:{np.std(v)}')


def article_test_util(
        sample_size=500,
        item_size=50,
        vi_class=VaeIRT,
        vi_class_kwargs=None,
        vi_fit_kwargs=None,
        random_class=RandomIrt2PL,
        random_class_kwargs=None,
        file_postfix=0,
        folder=None,
):
    model_name = random_class.name
    random_class_kwargs_ = {'sample_size': sample_size, 'item_size': item_size}
    if random_class_kwargs is not None:
        random_class_kwargs_.update(random_class_kwargs)
    torch.manual_seed(int(random.random() * 1000))
    r = random_class(**random_class_kwargs_)
    y = r.y
    x_feature = r.x.size(1)
    if folder is not None:
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder = f'{folder}/'
    else:
        folder = ''
    np.savetxt(
        f'{folder}{random_class.name or "data"}_sample_{sample_size}_item_{item_size}_dim_{x_feature}_{file_postfix}.txt',
        y.numpy()
    )
    np.savetxt(
        f'{folder}{random_class.name or "data"}_sample_{sample_size}_item_{item_size}_dim_{x_feature}_b_{file_postfix}.txt',
        r.b.numpy()
    )
    if hasattr(r, 'a'):
        np.savetxt(
            f'{folder}{random_class.name or "data"}_sample_{sample_size}_item_{item_size}_dim_{x_feature}_a_{file_postfix}.txt',
            r.a.T.numpy()
        )
    if hasattr(r, 'c'):
        np.savetxt(
            f'{folder}{random_class.name or "data"}_sample_{sample_size}_item_{item_size}_dim_{x_feature}_c_{file_postfix}.txt',
            r.c.T.numpy())
    if hasattr(r, 'd'):
        np.savetxt(
            f'{folder}{random_class.name or "data"}_sample_{sample_size}_item_{item_size}_dim_{x_feature}_d_{file_postfix}.txt',
            r.d.T.numpy()
        )
    vi_class_kwargs_ = {'data': y, 'model': model_name, 'subsample_size': 100, 'x_feature': x_feature}
    if vi_class_kwargs is not None:
        vi_class_kwargs_.update(vi_class_kwargs)
    vi_model = vi_class(**vi_class_kwargs_)
    vi_fit_kwargs_ = {'optim': Adam({'lr': 1e-2}), 'max_iter': 10000}
    if vi_fit_kwargs is not None:
        vi_fit_kwargs_.update(vi_fit_kwargs)
    vi_model.fit(random_instance=r, **vi_fit_kwargs_)
    rmse_dt = rmse_(item_size, model_name, r, x_feature)
    pyro.clear_param_store()
    return rmse_dt


def multiprocess_article_test_util(
        sample_size=500,
        item_size=50,
        vi_class=VaeIRT,
        vi_class_kwargs=None,
        vi_fit_kwargs=None,
        random_class=RandomIrt2PL,
        random_class_kwargs=None,
        start_idx=0,
        try_count=10,
        process_size=2,
        folder=None,
):
    pool = Pool(processes=process_size)
    res_lt = []
    for i in range(start_idx, start_idx + try_count):
        kwargs = {
            'sample_size': sample_size,
            'item_size': item_size,
            'vi_class': vi_class,
            'vi_class_kwargs': vi_class_kwargs,
            'vi_fit_kwargs': vi_fit_kwargs,
            'random_class': random_class,
            'random_class_kwargs': random_class_kwargs,
            'file_postfix': i,
            'folder': folder,
        }
        res = pool.apply_async(func=article_test_util, kwds=kwargs)
        res_lt.append(res)

    pool.close()
    pool.join()
    print_rmse(res_lt)


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

    def test_ai_100_dim_2pl(self):
        sample_size = 10000
        subsample_size = 100
        item_size = 500
        x_feature = 100
        random_instance = RandomMilIrt2PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=subsample_size, x_feature=x_feature)

        def optim(_, param_name):
            if param_name == 'a':
                return {'lr': 1e-2}
            if param_name == 'b':
                return {'lr': 1e-2}
            return {'lr': 1e-3}

        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': optim,
                                 'milestones': [
                                     # int(sample_size / subsample_size) * 50,
                                     int(sample_size / subsample_size) * 190
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 200), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    def test_ai_5_dim_3pl(self):
        sample_size = 10000
        subsample_size = 20
        item_size = 50
        x_feature = 10
        random_instance = RandomMilIrt3PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_3pl', subsample_size=subsample_size, x_feature=x_feature)

        def optim(_, param_name):
            if param_name == 'a':
                return {'lr': 1e-2}
            if param_name == 'b':
                return {'lr': 1e-2}
            if param_name == 'c':
                return {'lr': 1e-2}
            return {'lr': 1e-3}

        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': optim,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 190,
                                     # int(sample_size / subsample_size) * 400
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 200), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

    def test_ai_10_dim_4pl(self):
        sample_size = 10000
        subsample_size = 100
        item_size = 20
        x_feature = 10
        random_instance = RandomMilIrt4PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature)
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_4pl', subsample_size=subsample_size, x_feature=x_feature)

        def optim(_, param_name):
            if param_name == 'a':
                return {'lr': 1e-2}
            if param_name == 'b':
                return {'lr': 1e-2}
            return {'lr': 1e-3}

        scheduler = MultiStepLR({'optimizer': torch.optim.Adam,
                                 'optim_args': optim,
                                 'milestones': [
                                     int(sample_size / subsample_size) * 400,
                                 ],
                                 'gamma': 0.1,
                                 })
        model.fit(optim=scheduler, max_iter=int(sample_size / subsample_size * 500), random_instance=random_instance,
                  loss=Trace_ELBO(num_particles=1))

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
        sample_size = 1000
        subsample_size = 20
        item_size = 50
        x_feature = 2
        x_cov = torch.eye(x_feature)
        x_cov[0, 1] = x_cov[1, 0] = 0.7
        random_instance = RandomMilIrt2PL(sample_size=sample_size, item_size=item_size, x_feature=x_feature,
                                          x_cov=x_cov)
        y = random_instance.y
        model = VaeIRT(data=y, model='irt_2pl', subsample_size=subsample_size, x_feature=x_feature,
                       hidden_dim=64)

        def optim(_, param_name):
            if param_name == 'a':
                return {'lr': 1e-3}
            if param_name == 'b':
                return {'lr': 1e-3}
            return {'lr': 1e-3}

        model.fit(optim=Adam(optim), max_iter=int(sample_size / subsample_size * 1000), random_instance=random_instance,
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


class ArticleTest(TestCase):

    def test_4pl_bbvi_try_10_item_100_sample_1000(self):
        multiprocess_article_test_load_data_util(
            file_prefix='irt_4pl_1000',
            process_size=2,
            vi_class=VIRT,
            vi_class_kwargs=None,
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 1000},
            start_idx=0,
            try_count=10,
        )

    def test_2pl_bbvi_try_10_item_50_sample_100(self):
        multiprocess_article_test_load_data_util(
            file_prefix='irt_2pl_100',
            try_count=10,
            vi_class=VIRT,
            vi_class_kwargs=None,
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 10000},
            process_size=2,
            start_idx=0
        )

    def test_2pl_bbvi_try_10_item_100_sample_200(self):
        multiprocess_article_test_load_data_util(
            file_prefix='irt_2pl_200',
            vi_class=VIRT,
            vi_class_kwargs=None,
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 20000},
            start_idx=0,
            process_size=2
        )

    def test_2pl_bbvi_try_10_item_50_sample_500(self):
        multiprocess_article_test_load_data_util(
            file_prefix='irt_2pl_500',
            vi_class=VIRT,
            vi_class_kwargs=None,
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 20000},
            start_idx=0,
            process_size=2
        )

    # 均摊变分推断
    def test_2pl_ai_try_10_item_50_sample_100_dim_1(self):
        multiprocess_article_test_util(
            sample_size=100,
            item_size=50,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 20},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 1000},
            random_class=RandomIrt2PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
            folder='dt'
        )

    def test_2pl_ai_try_10_item_50_sample_200__dim_1(self):
        multiprocess_article_test_util(
            sample_size=200,
            item_size=50,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 20},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt2PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
            folder='dt'
        )

    def test_2pl_ai_try_10_item_100_sample_100_dim_1(self):
        multiprocess_article_test_util(
            sample_size=100,
            item_size=100,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 20},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 1000},
            random_class=RandomIrt2PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
            folder='dt'
        )

    def test_2pl_ai_try_10_item_100_sample_200_dim_1(self):
        multiprocess_article_test_util(
            sample_size=200,
            item_size=100,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 20},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt2PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
            folder='dt'
        )

    def test_2pl_ai_try_10_item_50_sample_500_dim_1(self):
        multiprocess_article_test_util(
            sample_size=500,
            item_size=50,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 1000},
            random_class=RandomIrt2PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2
        )

    def test_2pl_ai_try_10_item_100_sample_500_dim_1(self):
        multiprocess_article_test_util(
            sample_size=500,
            item_size=100,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 10000},
            random_class=RandomIrt2PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2
        )

    def test_3pl_ai_try_10_item_50_sample_500_dim_1(self):
        multiprocess_article_test_util(
            sample_size=500,
            item_size=50,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt3PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
        )

    def test_3pl_ai_try_10_item_100_sample_500_dim_1(self):
        multiprocess_article_test_util(
            sample_size=500,
            item_size=100,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt3PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
        )

    def test_3pl_ai_try_10_item_50_sample_1000_dim_1(self):
        multiprocess_article_test_util(
            sample_size=1000,
            item_size=50,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt3PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
        )

    def test_3pl_ai_try_10_item_100_sample_1000_dim_1(self):
        multiprocess_article_test_util(
            sample_size=1000,
            item_size=100,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt3PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
        )

    def test_4pl_ai_try_10_item_50_sample_1000_dim_1(self):
        multiprocess_article_test_util(
            sample_size=1000,
            item_size=50,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt4PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
        )

    def test_4pl_ai_try_10_item_100_sample_1000_dim_1(self):
        multiprocess_article_test_util(
            sample_size=1000,
            item_size=100,
            try_count=10,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 2000},
            random_class=RandomIrt4PL,
            random_class_kwargs=None,
            start_idx=0,
            process_size=2,
        )

    @staticmethod
    def test_2pl_ai_try_10_item_50_sample_1000_dim_2_optim(_, pn):
        if pn in ('a', 'b'):
            return {'lr': 1e-3}
        return {'lr': 1e-3}

    def test_2pl_ai_try_10_item_50_sample_1000_dim_2(self):
        multiprocess_article_test_util(
            sample_size=1000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': None},
            vi_fit_kwargs={'optim': Adam(self.test_2pl_ai_try_10_item_50_sample_1000_dim_2_optim), 'max_iter': 7000},
            random_class=RandomIrt2PL,
            random_class_kwargs={'x_feature': 2},
            start_idx=0,
            try_count=10,
            process_size=2,
        )

    def test_2pl_ai_try_10_item_50_sample_1000_dim_3(self):
        multiprocess_article_test_util(
            sample_size=1000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 50000},
            random_class=RandomIrt2PL,
            random_class_kwargs={'x_feature': 3},
            start_idx=0,
            try_count=10,
            process_size=2,
        )

    def test_2pl_ai_try_10_item_50_sample_5000_dim_3(self):
        multiprocess_article_test_util(
            sample_size=5000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 50000},
            random_class=RandomIrt2PL,
            random_class_kwargs={'x_feature': 3},
            start_idx=0,
            try_count=10,
            process_size=2,
        )

    def test_2pl_ai_try_10_item_50_sample_10000_dim_3(self):
        multiprocess_article_test_util(
            sample_size=10000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 50000},
            random_class=RandomIrt2PL,
            random_class_kwargs={'x_feature': 3},
            start_idx=0,
            try_count=10,
            process_size=2,
            folder='dt'
        )

    def test_2pl_ai_try_10_item_50_sample_5000_dim_5(self):
        multiprocess_article_test_util(
            sample_size=5000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 100000},
            random_class=RandomIrt2PL,
            random_class_kwargs={'x_feature': 5},
            start_idx=0,
            try_count=10,
            process_size=2,
            folder='dt',
        )

    def test_2pl_ai_try_10_item_50_sample_10000_dim_5(self):
        multiprocess_article_test_util(
            sample_size=10000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 100000},
            random_class=RandomIrt2PL,
            random_class_kwargs={'x_feature': 5},
            start_idx=0,
            try_count=10,
            process_size=2,
            folder='dt',
        )

    def test_3pl_ai_try_10_item_50_sample_10000_dim_5(self):

        multiprocess_article_test_util(
            sample_size=10000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-3}), 'max_iter': 100000},
            random_class=RandomIrt3PL,
            random_class_kwargs={'x_feature': 5},
            start_idx=0,
            try_count=10,
            process_size=2,
            folder='dt'
        )

    def test_3pl_mil_ai_try_10_item_50_sample_5000_dim_5(self):

        multiprocess_article_test_util(
            sample_size=5000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 5000},
            random_class=RandomMilIrt3PL,
            random_class_kwargs={'x_feature': 5},
            start_idx=0,
            try_count=10,
            process_size=2,
            folder='mil'
        )

    def test_4pl_mil_ai_try_10_item_50_sample_5000_dim_5(self):

        multiprocess_article_test_util(
            sample_size=5000,
            item_size=50,
            vi_class=VaeIRT,
            vi_class_kwargs={'subsample_size': 100},
            vi_fit_kwargs={'optim': Adam({'lr': 1e-2}), 'max_iter': 5000},
            random_class=RandomMilIrt4PL,
            random_class_kwargs={'x_feature': 5},
            start_idx=0,
            try_count=10,
            process_size=2,
            folder='mil'
        )