from unittest import TestCase

from pyro.optim import Adam
import torch

from vi import RandomIrt1PL, RandomIrt2PL, RandomIrt3PL, RandomIrt4PL, RandomDina, RandomDino, RandomHoDina, \
    VaeIRT, VIRT, VCDM, VHOCDM, VaeCDM


class TestMixin(object):

    def prepare_cuda(self):
        self.cuda = torch.cuda.is_available()
        print('cuda: {0}'.format(torch.cuda.is_available()))
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')


class IRTRandomMixin(object):

    def gen_sample(self, random_class, sample_size):
        random_instance = random_class(sample_size=sample_size)
        y = random_instance.y
        if self.cuda:
            y = y.cuda()
        return y, random_instance


class Irt1PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt1PL, 1000)
        irt = VIRT(data=y, model='irt_1pl', subsample_size=1000)
        irt.fit(random_instance=random_instance)

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt1PL, 100000)
        irt = VaeIRT(data=y, model='irt_1pl', subsample_size=100)
        irt.fit(random_instance=random_instance)


class Irt2PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt2PL, 1000)
        irt = VIRT(data=y, model='irt_2pl')
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt2PL, 100000)
        irt = VaeIRT(data=y, model='irt_2pl', subsample_size=100)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))


class Irt3PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt3PL, 1000)
        irt = VIRT(data=y, model='irt_3pl', subsample_size=1000)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}), max_iter=50000)

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt3PL, 100000)
        irt = VaeIRT(data=y, model='irt_3pl', subsample_size=100)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}), max_iter=50000)


class Irt4PLTestCase(TestCase, TestMixin, IRTRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, random_instance = self.gen_sample(RandomIrt4PL, 1000)
        irt = VIRT(data=y, model='irt_4pl', subsample_size=1000)
        irt.fit(random_instance=random_instance, max_iter=50000, optim=Adam({'lr': 5e-3}))

    def test_ai(self):
        y, random_instance = self.gen_sample(RandomIrt4PL, 100000)
        irt = VaeIRT(data=y, model='irt_4pl', subsample_size=100)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}), max_iter=50000)


class CDMRandomMixin(object):

    def gen_sample(self, random_class, sample_size):
        random_instance = random_class(sample_size=sample_size)
        y = random_instance.y
        attr = random_instance.attr
        if self.cuda:
            y = y.cuda()
            attr = attr.cuda()
        return y, attr, random_instance


class DinaTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, attr, random_instance = self.gen_sample(RandomDina, 1000)
        irt = VCDM(data=y, attr=attr, model='dina', subsample_size=1000)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, attr, random_instance = self.gen_sample(RandomDina, 100000)
        irt = VaeCDM(data=y, attr=attr, model='dina', subsample_size=100)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}))


class DinoTestCase(TestCase, TestMixin, CDMRandomMixin):

    def setUp(self):
        self.prepare_cuda()

    def test_bbvi(self):
        y, attr, random_instance = self.gen_sample(RandomDino, 1000)
        irt = VCDM(data=y, attr=attr, model='dino', subsample_size=1000)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))

    def test_ai(self):
        y, attr, random_instance = self.gen_sample(RandomDino, 100000)
        irt = VaeCDM(data=y, attr=attr, model='dino', subsample_size=100)
        irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}), max_iter=10000)
