# virt
virt是《变分推断在计量心理模型中的应用(variational inference for psychometrics model)》这篇论文的代码，这篇论文主要用于评职称，论文发表中。

斯坦福大学的项目[variational-item-response-theory-public](https://github.com/mhw32/variational-item-response-theory-public)结果无法重现，不知道是调参问题还是什么原因，并且斯坦福大学项目的问题还蛮多的，遂重新开发了基于变分推断的项目反应理论参数估计代码，并且加入了基于变分推断的认知诊断模型参数估计代码。

与其他（包括斯坦福的项目）基于变分推断的心理测量模型参数估计库（包）不同，本项目的模型不是全贝叶斯模型，是频率学派的心理测量模型。

# 与已有库（包）的区别
除了斯坦福大学的项目，其他基于变分推断的心理测量模型参数估计库（包）使用的算法是坐标下降变分推断，斯坦福大学项目的算法是他们自己号称的VIBO（他们的VIBO的证明有点问题啊），本项目使用的算法是黑盒变分推断和Amortized Variational Inference

# 本项目处理的模型
- 单参数IRT模型
- 双参数IRT模型
- 三参数IRT模型
- 四参数IRT模型
- DINA模型
- DINO模型
- HO-DINA模型

# 代码示例
生成IRT模型人工数据的方法
```python
def gen_irt_sample(random_class, sample_size):
    random_instance = random_class(sample_size=sample_size)
    y = random_instance.y
    return y, random_instance
```
生成认知诊断模型人工数据的方法
```python
def gen_cdm_sample(random_class, sample_size):
    random_instance = random_class(sample_size=sample_size)
    y = random_instance.y
    q = random_instance.q
    return y, q, random_instance
```
四参数IRT模型，基于黑盒变分推断
```python
from pyro.optim import Adam
from vi import VIRT, RandomIrt4PL

y, random_instance = gen_irt_sample(RandomIrt4PL, 1000)
irt = VIRT(data=y, model='irt_4pl', subsample_size=1000)
irt.fit(random_instance=random_instance, max_iter=50000, optim=Adam({'lr': 5e-3}))
```
四参数IRT模型，基于Amortized Variational Inference
```python
from pyro.optim import Adam
from vi import RandomIrt4PL, VaeIRT

y, random_instance = gen_irt_sample(RandomIrt4PL, 100000)
irt = VaeIRT(data=y, model='irt_4pl', subsample_size=100)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}), max_iter=50000)
```
DINA模型，基于黑盒变分推断
```python
from pyro.optim import Adam
from vi import RandomDina, VCDM

y, q, random_instance = gen_cdm_sample(RandomDina, 1000)
irt = VCDM(data=y, q=q, model='dina', subsample_size=1000)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))
```
DINA模型，基于Amortized Variational Inference
```python
from pyro.optim import Adam
from vi import RandomDina, VaeCDM

y, q, random_instance = gen_cdm_sample(RandomDina, 100000)
irt = VaeCDM(data=y, q=q, model='dina', subsample_size=100)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-2}))
```
DINA模型，基于离散潜变量黑盒变分推断
```python
from pyro.optim import Adam
from vi import RandomDina, VCCDM

y, q, random_instance = gen_cdm_sample(RandomDina, 1500)
irt = VCCDM(data=y, q=q, model='dina', subsample_size=1500)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))
```
DINA模型，基于离散潜变量Amortized Variational Inference
```python
from pyro.optim import Adam
from vi import RandomDina, VaeCCDM

y, q, random_instance = gen_cdm_sample(RandomDina, 100000)
irt = VaeCCDM(data=y, q=q, model='dina', subsample_size=100)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-2}))
```
HO-DINA模型，基于离散潜变量黑盒变分推断
```python
from pyro.optim import Adam
from vi import RandomHoDina, VCHoDina

y, q, random_instance = gen_cdm_sample(RandomHoDina, 1000)
irt = VCHoDina(data=y, q=q, subsample_size=1000)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 1e-1}))
```
HO-DINA模型，基于离散潜变量Amortized Variational Inference
```python
from pyro.optim import Adam
from vi import RandomHoDina, VaeCHoDina

y, q, random_instance = gen_cdm_sample(RandomHoDina, 100000)
irt = VaeCHoDina(data=y, q=q, subsample_size=100)
irt.fit(random_instance=random_instance, optim=Adam({'lr': 5e-3}))
```
更多测试用例详见测试文件