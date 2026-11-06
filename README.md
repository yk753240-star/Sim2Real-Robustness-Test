# Sim2Real-Robustness-Test
具身智能策略评估与 Sim2Real 鲁棒性测试
本项目基于 Python 与 MuJoCo 仿真环境，探究强化学习策略（PPO）在真实物理扰动下的性能衰减（Sim2Real Gap）。

技术栈
仿真引擎：MuJoCo
强化学习：Stable-Baselines3 (PPO)
数据处理：Pandas, Matplotlib

实验内容
在 Reacher-v5 环境中完成基础策略训练与可视化。
通过代码修改底层物理参数，引入质量突变（2.5倍）、传感器高斯噪声、以及动作执行延迟。
记录多回合评估数据并生成对比图表。

结果分析
标准环境平均奖励约为 -10.1，加入物理扰动后衰减至 -16.1。实验证明了底层硬件差异（Sim2Real）会导致策略严重失效。
