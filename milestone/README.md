### Tips
- 安装requirement的包，把数据集和噪音放这个文件里就行了
- 参数调整在config里面configs\data\train.yaml
- 测试轮数在configs\task\train_vm.yaml，目前是100可以先改1看看能不能跑

### 训练
```python
python run.py --task configs/task/train_vm.yaml --device cuda:0
```

### 去噪

- 到configs\task\predict_vm.yaml中改load_ckpt: 来调整用哪个模型

```python
python run.py --task configs/task/predict_vm.yaml --device cuda:0
```
