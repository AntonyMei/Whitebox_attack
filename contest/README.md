请先在git bash上运行下列代码, 安装比赛必要的包:

```bash
git clone https://github.com/thu-ml/ares
cd ares/
pip3 install -e .
```

https://thu-ml-ares.readthedocs.io/en/latest/ 是比赛的官方教程, 但是比较复杂, 可以先看一下 attacker 的 `__init__.py`.

之后, 请在链接 https://cloud.tsinghua.edu.cn/d/d5ff01ae54d847fc89a8/ 上更新并替换根目录下的`third_party` 文件夹.

进入 `./contest` 文件夹, 运行bash命令 `download_data.sh`. (可能需要vpn)

至此, 环境配置完毕.


初始化：采用ODI。但攻击一定轮数后，在当前对抗样本基础上重新进行ODI（试图模仿Ye Liu dalao的EWR，虽然没有揣度出他的真正方法），想法是在攻击过程中，定时调整扰动的方向。


Loss：采用CW loss。CW loss在许多攻击算法中都有应用，且相较于CE提升显著，而且很多黑盒攻击也是以CW loss作为loss oracle进行的攻击，总结一句话就是，CW，yyds。

auto PGD 使用二阶动量, 类似 adam

运行如下代码:

```bash
python run_attacks.py --attacks attacker4 --output ./tmp --models cifar10-pgd_at,cifar10-wideresnet_trades,cifar10-feature_scatter,cifar10-robust_overfitting,cifar10-rst,cifar10-fast_at,cifar10-at_he,cifar10-pre_training,cifar10-free_at,cifar10-awp,cifar10-hydra,cifar10-label_smoothing
```

