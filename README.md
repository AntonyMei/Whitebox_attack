请先在git bash上运行下列代码, 安装比赛必要的包:

```bash
git clone https://github.com/thu-ml/ares
cd ares/
pip3 install -e .
```

https://thu-ml-ares.readthedocs.io/en/latest/ 是比赛的官方教程, 但是比较复杂, 可以先看一下 attacker 的 `__init__.py`.

auto PGD 使用二阶动量, 类似 adam

