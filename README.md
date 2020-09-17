# vipsy
vipsy是《变分推断在教育测量模型中的应用(variational inference for psychometrics model)》这篇论文的代码，这篇论文主要用于评职称，论文发表中。

本项目受斯坦福大学的项目[variational-item-response-theory-public](https://github.com/mhw32/variational-item-response-theory-public)的激励。

恭喜斯坦福大学获得EDM2020最佳论文奖。

# 斯坦福大学项目存在的问题：
- 多维项目反应模型的参数估计精度严重失真，乃至不收敛，该项目只适合单维项目反应理论。
- 重参数方法不适用于认知诊断模型，不具备普适性

# 本项目与斯坦福项目的区别：
- 基于reinforce，而不是重参数方法。
- 让多维潜变量特质服从普通的多元正态分布，而不是对角多元正态分布。
- 可以选择让多元潜变量特质是否共享方差协方差矩阵。
- 神经网络生成的不仅有潜变量的均值和方差值，还有对角线以外的协方差值。
- 加入了认知诊断模型，包括DINA、HO-DINA等模型。
- 放弃了项目参数的随机性。
- 缺失数据的处理算法采用的是更简单但效果更好的方法。

# 与其他已有库（包）的区别
- 其他基于变分推断的心理测量模型参数估计库（包）使用的算法是坐标下降变分推断，本项目使用的算法是黑盒变分推断和Amortized Variational Inference。
- 多维项目反应模型的参数估计精度大大超越目前所有的开源与商业软件。
- 多维项目反应模型的运行时间和内存空间均大大超越MHRM算法。
- 测试用例展示了100个维度的项目反应理论参数估计
- 唯一的认知诊断

# 现存的缺点
- 认知诊断模型的参数估计方法实在拉胯，只适用属性模式较少的情况。
- 认知诊断模型的参数估计在运行时间上被EM算法吊打，百万级数据上EM算法也能应付，没有体现出变分推断算法的优势。
- 实验了Normalizing Flows，但没有发现什么优势，实乃不应该，这个需要再实验
- 学习速率与epoch的设置问题，前者太小后者就要大，从而导致运行时间过长

本项目虽有拾人牙慧的嫌疑，创新性不是太大，但作者碍于职称评定压力，不得不灌水发论文，实乃无奈之举。

# 本项目处理的模型
- 1-4参数单维IRT模型
- 1-4参数多维项目反应理论
- 缺失数据项目反应理论
- 验证性多维项目反应理论
- DINA模型
- DINO模型
- HO-DINA模型
- 缺失数据认知诊断模型

# 论文中的数据的测试用例
```shell script
$ python -m unittest test.ArticleTest
```

# 代码示例
四参数IRT模型，基于黑盒变分推断
```shell script
$ python -m unittest test.Irt4PLTestCase.test_bbvi 
```
四参数IRT模型，基于Amortized Variational Inference
```shell script
$ python -m unittest test.Irt4PLTestCase.test_ai
```
100个维度的多维项目反应理论模型，基于Amortized Variational Inference
```shell script
$ python -m unittest test.IrtMultiDimTestCase.test_ai_100_dim_2pl
```
缺失数据项目反应理论模型，缺失90%的数据（即仅有10%的数据有效）
```shell script
$ python -m unittest test.Irt2PLMissingTestCase.test_ai
```
4参数多维项目反应理论模型，5个维度
```shell script
$ python -m unittest test.IrtMultiDimTestCase.test_ai_10_dim_4pl
```
DINA模型，基于黑盒变分推断
```shell script
$ python -m unittest test.DinaTestCase.test_bbvi
```
DINA模型，基于Amortized Variational Inference
```shell script
$ python -m unittest test.DinaTestCase.test_ai
```
DINA模型，基于离散潜变量黑盒变分推断
```shell script
$ python -m unittest test.PaDinaTestCase.test_bbvi
```
DINA模型，基于离散潜变量Amortized Variational Inference
```shell script
$ python -m unittest test.PaDinaTestCase.test_ai
```
HO-DINA模型，基于离散潜变量黑盒变分推断
```shell script
$ python -m unittest test.PaHoDinaTestCase.test_bbvi
```
HO-DINA模型，基于离散潜变量Amortized Variational Inference
```shell script
$ python -m unittest test.PaHoDinaTestCase.test_ai
```
更多测试用例详见[测试文件](https://github.com/inuyasha2012/virt/blob/master/test.py)