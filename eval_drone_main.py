# eval_drone_main.py
# ==============================================================================
# 作用：主程序入口 (Main Loop)
# 职责：
#   1. 初始化 Isaac Sim 环境 (SimulationApp)
#   2. 加载数据集 (LeRobotDatasetLoader) 或 Mock 数据 (MockDatasetLoader)
#   3. 遍历每个 Episode，动态加载对应的 USD 场景
#   4. 调用 DroneRunner 执行单次任务
# 使用方法： /isaac-sim/python.sh eval_drone_main.py --dataset_path /path/to/dataset
# ⚠️需要先打开server端服务再运行本文件
# ==============================================================================
from isaacsim import SimulationApp

# 在所有 import 之前启动 App
simulation_app = SimulationApp({"headless": True})

import argparse
import os
import pandas as pd
import numpy as np
from omni.isaac.core import World
import glob
import json
from omni.isaac.core.utils.stage import add_reference_to_stage, clear_stage

# 导入core中的执行逻辑
from eval_drone_core import DroneRunner

# ==============================================================================
# [MODEL INTERFACE] 真实数据集加载器
# ------------------------------------------------------------------------------
# 作用：读取 LeRobot/Habitat 生成的真实数据，包括场景ID、指令、起点、终点、GT轨迹。
# ⚠️这是旧版读取数据代码，适配lerobot格式的
# ==============================================================================
# class LeRobotDatasetLoader:
#     def __init__(self, input_path):
#         self.traj_folders = []
#         # 判断 input_path 是“单个轨迹”还是“数据集根目录”
#         # 如果目录下直接有 meta/episodes.jsonl，说明它就是一个具体的 trajectory 文件夹
#         if os.path.exists(os.path.join(input_path, "meta", "episodes.jsonl")):
#             self.traj_folders = [input_path]
#             print(f"Loading single trajectory: {input_path}")
#         else:
#             # 否则假设它是根目录，递归搜索所有 trajectory_* 文件夹
#             print(f"Searching for trajectories in: {input_path}")
#             self.traj_folders = sorted(glob.glob(os.path.join(input_path, "**", "trajectory_*"), recursive=True))
            
#         print(f"Found {len(self.traj_folders)} episodes.")
        
#     def __len__(self):
#         return len(self.traj_folders)

#     def get_episode(self, index):
#         # 读取单个 Episode 的所有元数据
#         traj_folder = self.traj_folders[index]
#         scene_id = os.path.basename(os.path.dirname(traj_folder))
#         episode_id = os.path.basename(traj_folder)

#         # === 1. 解析自然语言指令 (按照Internnva应该是从meta/episodes.jsonl读取) ===
#         instruction = "Go to goal" # 默认值
#         meta_path = os.path.join(traj_folder, "meta", "episodes.jsonl")     
#         if os.path.exists(meta_path):
#             try:
#                 with open(meta_path, 'r') as f:
#                     # 读取第一行 JSON
#                     meta_root = json.loads(f.readline())
#                     # [关键修改] "tasks" 里面存的是 JSON 字符串，需要 json.loads 两次
#                     if "tasks" in meta_root and len(meta_root["tasks"]) > 0:
#                         task_str = meta_root["tasks"][0] # 这是一个字符串
#                         task_data = json.loads(task_str) # 解析成字典
#                         instruction = task_data.get("sum_instruction", instruction)
#             except Exception as e:
#                 print(f"⚠️ Error reading instruction: {e}")

#         # === 2. 解析 Ground Truth (GT) 轨迹 ===
#         # 主要用于提取 Start/Goal 坐标，以及画参考线
#         # 这一部分目前是读取action然后提取里面的坐标轨迹，将这条轨迹作为GT的，如果有别的方法改在这里就可以
#         parquet_path = os.path.join(traj_folder, "data/chunk-000/episode_000000.parquet")
#         print(f"DEBUG: 正在寻找 GT 文件: {parquet_path}")
#         start_pos = np.array([0.0, 0.0, 1.5]) # 默认值
#         goal_pos = np.array([5.0, 5.0, 1.5])
#         gt_trajectory = None

#         if os.path.exists(parquet_path):
#             try:
#                 df = pd.read_parquet(parquet_path)
#                 target_col = 'action'
#                 extrinsics = np.stack(df['action'].to_numpy())
#                 # print("\n--- [DEBUG GT DATA] ---")
#                 # print(f"Raw Extrinsic Shape: {extrinsics[0].shape}")
#                 # print(f"Raw Extrinsic Data (First Frame): {extrinsics[0]}")
                
#                 if target_col in df.columns:
#                     poses = np.stack(df[target_col].to_numpy())
#                     gt_temp = []
#                     for pose in poses:
#                         # 适配不同格式的矩阵，Internnva是4×4的
#                         if len(pose) == 4: # 4x4 matrix
#                             rx, ry, rz = pose[0][3], pose[1][3], pose[2][3]
#                         elif len(pose) == 16: # Flattened
#                             rx, ry, rz = pose[3], pose[7], pose[11]
#                         else: continue

#                         # 重要！！！！！坐标系转换：Habitat (Y-up) -> Isaac (Z-up)
#                         # 具体映射关系需根据实际数据调整，目前为: x=y, y=x, z=z
#                         # 这里目前我试的是这样子，但是可能导致左右手系变化？具体数据集具体试一下
#                         ix = ry
#                         iy = rx
#                         iz = rz 
#                         gt_temp.append([ix, iy, iz])

#                     if len(gt_temp) > 0:
#                         gt_raw = np.array(gt_temp)
#                         raw_start = gt_raw[0]
#                         # 相对位移归一化：将起点强制对齐到 (0,0,1.5)
#                         offset = np.array([0.0, 0.0, 1.5]) - raw_start
#                         # 应用偏移到所有点
#                         gt_trajectory = gt_raw + offset                      
#                         start_pos = gt_trajectory[0] 
#                         goal_pos = gt_trajectory[-1]
                        
#                         # 测试用print
#                         # print(f"✅ Loaded & Normalized GT. Points: {len(gt_trajectory)}")
#                         # print(f"   Original Start: {raw_start}")
#                         # print(f"   New Start (Warehouse): {start_pos}")
#                         # print(f"   Offset Applied: {offset}")

#             except Exception as e:
#                 print(f"⚠️ Error reading parquet: {e}")

#         return {
#             "episode_id": episode_id,
#             "scene_id": scene_id,
#             "instruction": instruction, # 放入解析好的指令
#             "start_pos": start_pos,
#             "goal_pos": goal_pos,
#             "gt_trajectory": gt_trajectory
#         }


# ==============================================================================
# [MODEL INTERFACE] 真实数据集加载器 (适配新版 Dataset 结构)
# ⚠️因为数据集格式做了变化，故重写了一版读取数据的代码，但是没有真实代码测试过，一些参数拿取位置之后可以按需调整
# ------------------------------------------------------------------------------
# 结构参考:
# Dataset_Root/
#   ├── scene_description0/
#   │   ├── episodes_extras.parquet (外参/内参)
#   │   ├── pointcloud.ply
#   │   ├── meta/
#   │   │   └── episodes.jsonl (索引信息)
#   │   |── data/
#   │   |   └── chunk-000/
#   │   |       └── episode_000000.parquet (轨迹数据)
#   |   |—— images
#   │       └── chunk-000/
#   │           |── observation.depth.front
#   |           |   └── episodes_000000
#   |           |       └── 00000.png
#   │           └── observation.depth.left
#   └── scene_description1/ ...
# ==============================================================================

class LeRobotDatasetLoader:
    def __init__(self, input_path):
        """
        input_path: 数据集根目录 (包含多个 scene_description 文件夹)
        """
        self.dataset_root = input_path
        self.episodes_index = [] # 扁平化的索引列表，存每个 episode 的元信息

        # 1. 扫描所有场景文件夹
        # 假设场景文件夹以 scene_description 开头，或者直接扫描所有子文件夹
        scene_folders = sorted([f for f in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(f)])
        
        print(f"Found {len(scene_folders)} scenes in {input_path}")

        for scene_path in scene_folders:
            scene_id = os.path.basename(scene_path)
            meta_path = os.path.join(scene_path, "meta", "episodes.jsonl")
            extras_path = os.path.join(scene_path, "episodes_extras.parquet")

            if not os.path.exists(meta_path):
                continue

            # 2. 读取每个场景的 meta/episodes.jsonl 来建立索引
            try:
                # 读取 episodes.jsonl (通常每一行是一个 episode 的 json)
                # LeRobot 标准格式中，episodes.jsonl 包含 "episode_index" 和 "chunk" 信息
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        ep_meta = json.loads(line)
                        
                        # 提取关键信息用于后续定位文件
                        episode_id = ep_meta.get("episode_index")
                        # 兼容处理：有些版本 chunk 可能是根据 ID 算的，有些直接写在 meta 里
                        # 如果没有 chunk 字段，默认认为每 1000 个是一个 chunk (LeRobot常见默认值)
                        chunk_id = ep_meta.get("chunk", episode_id // 1000 if episode_id is not None else 0)
                        
                        # 解析指令 (提前解析，存入索引，避免 get_item 时重复 IO)
                        instruction = "Go to goal"
                        if "tasks" in ep_meta and len(ep_meta["tasks"]) > 0:
                            # 注意：根据你的描述，tasks 里面可能是字符串需要二次解析
                            task_raw = ep_meta["tasks"][0]
                            if isinstance(task_raw, str):
                                try:
                                    task_data = json.loads(task_raw)
                                    instruction = task_data.get("sum_instruction", instruction)
                                except:
                                    instruction = task_raw # 如果不是json字符串，直接用
                            elif isinstance(task_raw, dict):
                                instruction = task_raw.get("sum_instruction", instruction)

                        self.episodes_index.append({
                            "scene_id": scene_id,
                            "scene_path": scene_path,
                            "episode_id": episode_id,
                            "chunk_id": chunk_id,
                            "instruction": instruction,
                            "extras_path": extras_path # 存下 extras 路径备用
                        })
            except Exception as e:
                print(f"⚠️ Error indexing scene {scene_id}: {e}")

        print(f"Total episodes indexed: {len(self.episodes_index)}")

    def __len__(self):
        return len(self.episodes_index)

    def get_episode(self, index):
        # 获取当前 episode 的索引信息
        info = self.episodes_index[index]
        
        scene_path = info["scene_path"]
        episode_id = info["episode_id"]
        chunk_id = info["chunk_id"]
        instruction = info["instruction"]
        
        # === 1. 构建 Parquet 文件路径 (动态 Chunk) ===
        # 格式通常为: data/chunk-000/episode_000000.parquet
        chunk_str = f"chunk-{chunk_id:03d}"
        ep_str = f"episode_{episode_id:06d}.parquet"
        parquet_path = os.path.join(scene_path, "data", chunk_str, ep_str)

        # 默认值
        start_pos = np.array([0.0, 0.0, 1.5])
        goal_pos = np.array([5.0, 5.0, 1.5])
        gt_trajectory = None

        # === 2. 读取 Extras (外参矩阵) [根据图片新增] ===
        # 如果需要将相机坐标系转换到机体坐标系，需要这个矩阵
        extrinsic_matrix = None # 默认为 Identity
        if os.path.exists(info["extras_path"]):
            try:
                # episodes_extras.parquet 通常包含每一条轨迹的额外信息
                # 需要根据 episode_index 筛选
                # 假设 extras 文件里有 "episode_index" 列，或者它包含所有 episode 的信息
                df_extras = pd.read_parquet(info["extras_path"])
                
                # 尝试找到当前 episode 对应的行
                if "episode_index" in df_extras.columns:
                    row = df_extras[df_extras["episode_index"] == episode_id]
                    if not row.empty:
                        # 提取 Extrinsic_front (4x4 矩阵)
                        # 注意：parquet存的一般是 flatten 的 list 或 numpy array
                        if "Extrinsic_front" in row.columns:
                            ext_val = row.iloc[0]["Extrinsic_front"]
                            extrinsic_matrix = np.array(ext_val).reshape(4, 4)
            except Exception as e:
                print(f"⚠️ Error reading extras: {e}")

        # === 3. 解析 GT 轨迹 ===
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                
                # 假设 'action' 列存放的是位姿或轨迹点
                target_col = 'action' # 或者是 'observation.state'，根据具体数据确认
                
                if target_col in df.columns:
                    # 提取数据并堆叠
                    raw_data = np.stack(df[target_col].to_numpy())
                    
                    gt_temp = []
                    for item in raw_data:
                        # 提取 x, y, z
                        # 情况 A: 只有位置 (3,)
                        if item.size == 3:
                            gt_temp.append(item)
                        # 情况 B: 4x4 矩阵
                        elif item.size == 16: 
                            mat = item.reshape(4, 4) if item.ndim == 1 else item
                            # 提取位移部分
                            gt_temp.append(mat[:3, 3]) 
                        # 情况 C: (6,) 或 (7,) 含旋转
                        elif item.size >= 6:
                            gt_temp.append(item[:3])

                    if len(gt_temp) > 0:
                        gt_trajectory = np.array(gt_temp)

                        # --- 坐标系转换逻辑 ---
                        # 如果需要应用 extrinsic (Camera -> Body)，在这里乘矩阵
                        # 图片提到: "Extrinsic_front": 4x4 矩阵, 存相机坐标系到机体坐标系转换矩阵
                        # 如果 gt_trajectory 是基于相机的，则需要转换：
                        if extrinsic_matrix is not None:
                            # 齐次坐标转换
                            ones = np.ones((len(gt_trajectory), 1))
                            points_homo = np.hstack([gt_trajectory, ones]) # (N, 4)
                            # Points_body = T_cam2body * Points_cam
                            # 注意转置，取决于矩阵存储方式，通常是 (T @ P.T).T
                            transformed = (extrinsic_matrix @ points_homo.T).T 
                            gt_trajectory = transformed[:, :3]

                        # --- 归一化逻辑 (保持你原有的逻辑) ---
                        # 机体坐标系定义: +x forward, +y left, +z up
                        raw_start = gt_trajectory[0]
                        # 强制对齐到 (0,0,1.5) 或其他高度
                        offset = np.array([0.0, 0.0, 1.5]) - raw_start
                        gt_trajectory = gt_trajectory + offset
                        
                        start_pos = gt_trajectory[0]
                        goal_pos = gt_trajectory[-1]

            except Exception as e:
                print(f"⚠️ Error reading parquet {parquet_path}: {e}")
        else:
            print(f"❌ Parquet file not found: {parquet_path}")

        return {
            "episode_id": f"{info['scene_id']}_ep{episode_id}", # 唯一ID
            "scene_id": info["scene_id"],
            "instruction": instruction,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "gt_trajectory": gt_trajectory
        }

# ==============================================================================
# [TEST ONLY] Mock 数据加载器
# ------------------------------------------------------------------------------
# 作用：仅在没有提供 dataset_path 时使用。用于快速测试环境是否跑通并人为给出轨迹测试
# 接模型后：这部分代码可忽略或删除。
# ==============================================================================
class MockDatasetLoader:
    def __init__(self):
        print("⚠️ 警告: 未提供数据集路径，进入测试模式 (Mock Mode)")
        
    def __len__(self):
        return 1 # 只跑一条测试

    def get_episode(self, index):
        # 返回我们之前验证通过的固定参数
        return {
            # "episode_id": 999,
            # "episode_id": "cluttered_easy",
            # "episode_id": "scenes_home",
            "episode_id": "scenes_commercial_new",
            # "scene_id": "warehouse", # 对应 assets/scenes/warehouse.usd
            # "scene_id": "cluttered_easy/easy_0/cluttered-0",
            # "scene_id": "internscenes_home/scenes_home/MVUCSQAKTKJ5EAABAAAAABI8_usd/start_result_navigation",
            "scene_id": "internscenes_commercial/scenes_commercial/MV5M25QKTKJZ2AABAAAAAAI8_usd/start_result_navigation",
            "instruction": "Fly to the red ball (Test)",
            # "start_pos": np.array([0.0, -5.0, 1.5]), #这俩是warehouse.usd的
            # "goal_pos": np.array([1.5, -5.0, 1.5])
            "start_pos": np.array([0.0, 0.0, 1.5]),  #这俩是新测试的
            "goal_pos": np.array([1.5, 0.0, 1.5])
        }

def main():
    parser = argparse.ArgumentParser()
    # [MODEL CONFIG] 运行时请指定这个参数指向真实数据集
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to LeRobot dataset root")
    parser.add_argument("--drone_usd", type=str, default="./assets/robots/quadcopter.usd")
    parser.add_argument("--scene_root", type=str, default="./assets/scenes", help="Folder containing scene USDs")
    # [MODEL CONFIG] 模型推理 Server 的地址
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:9000/act")
    args = parser.parse_args()

    # 1. 初始化环境
    physics_dt = 1.0/60.0
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)
    
    # 2. 初始化执行器 Runner
    runner = DroneRunner(world, args.server_url)
    
    # 3. 选择加载器
    if args.dataset_path and os.path.exists(args.dataset_path):
        dataset = LeRobotDatasetLoader(args.dataset_path)
    else:
        dataset = MockDatasetLoader()

    print(f"Total episodes to run: {len(dataset)}")

    current_scene_id = None

    # 4. 循环测试
    for i in range(len(dataset)):
        data = dataset.get_episode(i)
        scene_id = data['scene_id']
        episode_id = data['episode_id']
        
        print(f"\n--- Processing Episode {episode_id} (Scene: {scene_id}) ---")

        # --- [SCENE LOADING] 场景动态加载逻辑 ---
        # 目前考虑的是如果下一条数据还是同一个场景，就不重新加载，提升速度
        if scene_id != current_scene_id:
            usd_path = os.path.join(args.scene_root, f"{scene_id}.usd")
            
            # 如果是测试模式，且找不到该文件，回退到默认 warehouse
            # 只有在测试模式下才这么做，为了保证能跑通，真实接模型可以删掉warehouse这部分回退
            if not os.path.exists(usd_path):
                print(f"Warning: Scene {usd_path} not found. Trying fallback...")
                fallback_path = "./assets/scenes/warehouse.usd"
                if os.path.exists(fallback_path):
                    usd_path = fallback_path
                else:
                    print("Error: Fallback scene not found either!")
                    continue

            print(f"Loading Scene: {usd_path}")
            # 使用 Reference 方式加载场景，挂载到 /World/Map 节点下
            add_reference_to_stage(usd_path=usd_path, prim_path="/World/Map")
            current_scene_id = scene_id
        
        # --- [DRONE SETUP] 配置无人机和相机 ---
        # 根据 start/goal 计算出全局地图尺寸，返回给主循环
        map_size_info = runner.setup_drone(args.drone_usd, data['start_pos'], data['goal_pos'])
        
        # --- [RUN] 执行任务 ---
        # 传入 map_size_info，确保画图Recorder能正确初始化
        # 其中超参数K是从模型得到action后执行前K步，is_stop是判决模型输出小于is_stop的时候认为停止
        runner.run_episode(data, map_size_info, max_steps=500, K=5, is_stop=1e-5)

    print("All done!")
    simulation_app.close()

if __name__ == "__main__":
    main()