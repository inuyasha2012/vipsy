from unittest import TestCase

import pyro
import torch

from vi import RandomIrt1PL, RandomIrt2PL, RandomIrt3PL, RandomIrt4PL, RandomDina, RandomDino, RandomHoDina, \
    VaeIRT, VIRT, VCDM, VHOCDM


class Irt1PLTestCase(TestCase):

    def setUp(self) -> None:
        print('cuda: {0}'.format(torch.cuda.is_available()))
        random_irt = RandomIrt1PL()
        self.y = random_irt.y
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.y = self.y.cuda()

    def test_bbvi_irt(self):
        irt = VIRT(data=self.y, model='irt_1pl', subsample_size=1000)
        irt.fit()

    def test_ai_irt(self):
        irt = VaeIRT(data=self.y, model='irt_1pl', subsample_size=1000)
        irt.fit()


class Irt2PLTestCase(TestCase):

    def setUp(self) -> None:
        print('cuda: {0}'.format(torch.cuda.is_available()))
        random_irt = RandomIrt2PL()
        self.y = random_irt.y
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.y = self.y.cuda()

    def test_bbvi_2pl(self):
        irt = VIRT(data=self.y, model='irt_2pl', subsample_size=1000)
        irt.fit()

    def test_ai_2pl(self):
        irt = VIRT(data=self.y, model='irt_2pl', subsample_size=1000)
        irt.fit()


class Irt3PLTestCase(TestCase):

    def setUp(self) -> None:
        print('cuda: {0}'.format(torch.cuda.is_available()))
        random_irt = RandomIrt3PL()
        self.y = random_irt.y
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.y = self.y.cuda()

    def test_bbvi_3pl(self):
        irt = VIRT(data=self.y, model='irt_3pl', subsample_size=1000)
        irt.fit()

    def test_ai_3pl(self):
        irt = VIRT(data=self.y, model='irt_3pl', subsample_size=1000)
        irt.fit()


class Irt4PLTestCase(TestCase):

    def setUp(self) -> None:
        print('cuda: {0}'.format(torch.cuda.is_available()))
        random_irt = RandomIrt4PL()
        self.y = random_irt.y
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.y = self.y.cuda()

    def test_bbvi_2pl(self):
        irt = VIRT(data=self.y, model='irt_4pl', subsample_size=1000)
        irt.fit()

    def test_ai_2pl(self):
        irt = VaeIRT(data=self.y, model='irt_4pl', subsample_size=1000)
        irt.fit()
