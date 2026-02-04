# Isaac Sim Drone Navigation Benchmark

基于 Isaac Sim 的无人机视觉导航评测接口。支持加载 LeRobot 数据集，包含 Sim-to-Real 的标准接口设计。

## 📂 项目结构

- **`eval_drone_main.py`**: 程序入口。负责加载数据集 (LeRobot/Habitat) 和场景 (USD)，调度评测循环。
- **`eval_drone_core.py`**: 核心库。包含 PID 控制器、Benchmark 记录器 (绘图/计算 SPL 指标) 和 Runner 执行逻辑。
- **`drone_server.py`**: 策略服务器 (Flask)。支持 Mock 几何导航和深度学习模型推理接口。

## 🛠️ 环境配置 (Environment Setup)

本项目基于 **NVIDIA Isaac Sim 5.0** 开发。为了确保可复现性，建议使用 Docker 容器运行。

### 1. 拉取镜像
本项目使用的 Docker 镜像名称为 `nvcr.io/nvidia/isaac-sim:5.0.0`，直接拉取镜像即可。
目前已有Docker `isaacsim_benchmark`

### 2. 依赖安装
   ```bash
   docker exec -it your_docker_name bash

   /isaac-sim/python.sh -m pip install opencv-python matplotlib scipy flask json_numpy
   ```

## 📦 资产配置 (Assets Setup)

由于资产文件 (USD/Textures) 体积过大，未包含在代码仓库中。请按照以下步骤配置：

1. **创建目录结构**：
   在项目根目录下手动创建以下文件夹：
   ```bash
   mkdir -p assets/scenes
   mkdir -p assets/robots

2. **下载资产**：
scenes资产在pro6000仿真服务器的/mnt/data0/datasets/Scene-N1/n1_eval_scenes下，按需下载即可
对于仅测试，scenes资产可以通过已有的文件下载，注意改一下文件的存放路径
   ```bash
   python debug/download_assets.py
请联系我获取 robots 的资产压缩包，将无人机 .usd 文件解压到 robots 目录中。


**最终的文件目录结构应严格如下所示**：
   ```text
   assets/
   ├── robots/
   │   ├── dronedemo.usd
   │   └── quadcopter.usd
   └── scenes/
       ├── cluttered_easy/
       ├── internscenes_commercial/
       ├── internscenes_home/
       ├── Materials/
       ├── SkyTexture/
       └── warehouse.usd
   ```

3. **数据集**：
目前数据集没有整理和确定下来建议是建一个文件夹专门存放。具体数据集之后更新
   ```bash
   mkdir -p data
   ```

## 🚀 快速开始

### 1. 启动策略服务器
在单独的终端运行：
```bash
python drone_server.py
```

### 2. 运行评测
```bash
# 运行 Mock 测试 (无需数据集)
python eval_drone_main.py

# 运行真实数据集
python eval_drone_main.py --dataset_path /path/to/lerobot_dataset
