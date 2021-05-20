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





auto PGD 使用二阶动量, 类似 adam

