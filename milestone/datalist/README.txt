该目录已包含赛题提供的数据划分清单：
- train.txt：训练集模型相对路径
- validate.txt：验证集模型相对路径
- test.txt：测试集模型相对路径

默认配置会直接读取这三个文件，无需再次运行 scripts/build_datalists.py。
只有更换数据集或希望重新划分训练/验证集时，才需要运行该脚本。
