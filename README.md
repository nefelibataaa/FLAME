FLAME 第三章算法核心实现

CodeLlama 训练
---------------------
GPU机器请使用已经安装兼容驱动的 CUDA PyTorch 的独立环境，安装 requirements.txt。

  MODEL=modelpath \\
    DATA=datapath \\
    PYTHON=python bash run_train.sh --epochs 1 --max-length 128 --remove-layers 23

上述命令使用当前CUDA环境和人工数据，仅用于跑通流程。
run_train.sh 默认使用 GPU 0、1，通过 torchrun 启动两进程 DDP；NPROC_PER_NODE=1 可单卡运行。
每卡保留一份完整模型，同步LoRA和分类头梯度；不是把权重分到两张卡上。
每卡batch_size=1时，全局batch_size=2，不保证与单卡batch_size=1参数轨迹相同。
DistributedSampler在样本数不能整除卡数时补齐少量样本，日志区分samples与unique_samples。
评估、剪枝与模型保存仅由rank0执行，不存在两个rank同时覆盖输出文件的问题。

后台启动同样的两卡测试，内部已用nohup，无需再加 &：
  bash run_train_background.sh

默认模型为上面的CodeLlama路径，DATA为本项目data/smoke_code.jsonl；默认1轮、128 tokens、不剪枝。
启动后打印PID和日志位置：本项目train_<时间戳>_<启动进程号>.log，PID写入同名.log.pid。
可用LOG_FILE=/绝对路径/train.log覆盖日志位置，已存在的日志会拒绝覆盖。
参数如 --epochs 3 可追加在run_train_background.sh后覆盖测试默认值。
后台启动只表示进程已拉起，不保证预检/训练成功，需tail -f查看日志。

数据格式为JSONL，每条至少包含：
  {"id":"sample-001","code":"print(1)","label":0,"language":"python"}
  label必须为整数0/1：0良性、1恶意；language取python/javascript。
  会拒绝重复ID/精确重复代码；按language+label分层随机划分约7:1:2，保存完整划分清单。
  每个分层组至少10条。
