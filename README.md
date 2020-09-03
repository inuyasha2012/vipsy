# vipsy
vipsy是《变分推断在计量心理模型中的应用(variational inference for psychometrics model)》这篇论文的代码，这篇论文主要用于评职称，论文发表中。

斯坦福大学的项目[variational-item-response-theory-public](https://github.com/mhw32/variational-item-response-theory-public)结果无法重现，不知道是调参问题还是什么原因，并且斯坦福大学项目的问题还蛮多的，遂重新开发了基于变分推断的项目反应理论参数估计代码，并且加入了基于变分推断的认知诊断模型参数估计代码。

本项目虽有拾人牙慧的嫌疑，但作者碍于职称评定压力，不得不灌水发论文，实乃无奈之举，当然，本项目和之前的诸多变分推断R包和python库，还是有区别的。

与其他（包括斯坦福的项目）基于变分推断的心理测量模型参数估计库（包）不同，本项目的模型不是全贝叶斯模型，是频率学派的心理测量模型。

# 与已有库（包）的区别
- 除了斯坦福大学的项目，其他基于变分推断的心理测量模型参数估计库（包）使用的算法是坐标下降变分推断，斯坦福大学项目的算法是他们自己号称的VIBO（他们的VIBO的有点问题啊），本项目使用的算法是黑盒变分推断和Amortized Variational Inference
- 多维项目反应模型的参数估计精度完爆斯坦福大学项目
- 多维项目反应模型的参数估计精度和运行时间完爆MHRM算法（mirt包实现）
- 测试用例展示了100个维度的项目反应理论

# 本项目处理的模型
- 1-4参数单维IRT模型
- 1-4参数多维项目反应理论
- 缺失数据项目反应理论
- 验证性多维项目反应理论
- DINA模型
- DINO模型
- HO-DINA模型
- 缺失数据认知诊断模型

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
python -m unittest test.PaHoDinaTestCase.test_bbvi
```
HO-DINA模型，基于离散潜变量Amortized Variational Inference
```shell script
python -m unittest test.PaHoDinaTestCase.test_ai
```
更多测试用例详见[测试文件](https://github.com/inuyasha2012/virt/blob/master/test.py)