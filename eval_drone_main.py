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
import datetime

# 导入core中的执行逻辑
from eval_drone_core import DroneRunner

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
        self.episodes_index = [] 
        
        print(f"\n🔍 [Dataset] 开始扫描数据集根目录: {input_path}")
        
        # 1. 扫描所有场景文件夹 (形如 scene_description0, scene_description1 ...)
        scene_folders = sorted([f for f in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(f)])
        print(f"✅ [Dataset] 共发现 {len(scene_folders)} 个场景文件夹。")

        for scene_path in scene_folders:
            scene_id = os.path.basename(scene_path)
            scene_id = "hospital/hospital" #也可以在这里改场景
            meta_path = os.path.join(scene_path, "meta", "episodes.jsonl")
            extras_path = os.path.join(scene_path, "episodes_extras.parquet")

            if not os.path.exists(meta_path):
                print(f"⚠️ [Dataset] 场景 {scene_id} 缺少 meta/episodes.jsonl，已跳过。")
                continue

            # 2. 读取 meta 建立索引
            episodes_in_scene = 0
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        ep_meta = json.loads(line)
                        episode_id = ep_meta.get("episode_index")
                        chunk_id = ep_meta.get("chunk", episode_id // 1000 if episode_id is not None else 0)
                        
                        # 解析指令
                        instruction = "Go to goal"
                        if "tasks" in ep_meta and len(ep_meta["tasks"]) > 0:
                            task_raw = ep_meta["tasks"][0]
                            if isinstance(task_raw, str):
                                try:
                                    task_data = json.loads(task_raw)
                                    instruction = task_data.get("sum_instruction", instruction)
                                except:
                                    instruction = task_raw 
                            elif isinstance(task_raw, dict):
                                instruction = task_raw.get("sum_instruction", instruction)

                        self.episodes_index.append({
                            "scene_id": scene_id,
                            "scene_path": scene_path,
                            "episode_id": episode_id,
                            "chunk_id": chunk_id,
                            "instruction": instruction,
                            "extras_path": extras_path 
                        })
                        episodes_in_scene += 1
                print(f"✅ [Dataset] 场景 {scene_id} 索引建立完成，共 {episodes_in_scene} 条轨迹。")
            except Exception as e:
                print(f"❌ [Dataset] 场景 {scene_id} 索引建立失败: {e}")

        print(f"🎉 [Dataset] 数据集加载完毕！总计载入 {len(self.episodes_index)} 个有效 Episode。\n")

    def __len__(self):
        return len(self.episodes_index)

    def get_episode(self, index):
        info = self.episodes_index[index]
        scene_path = info["scene_path"]
        episode_id = info["episode_id"]
        chunk_id = info["chunk_id"]
        instruction = info["instruction"]
        
        print(f"\n" + "="*50)
        print(f"🚀 [Dataset Info] 正在准备 Episode: {episode_id} | Scene: {info['scene_id']}")
        print(f"🗣️ [Instruction] {instruction}")
        
        # 路径拼接
        chunk_str = f"chunk-{chunk_id:03d}"
        ep_str = f"episode_{episode_id:06d}.parquet"
        parquet_path = os.path.join(scene_path, "data", chunk_str, ep_str)

        start_pos = np.array([0.0, 0.0, 1.5])
        goal_pos = np.array([5.0, 5.0, 1.5])
        gt_trajectory = None
        extrinsic_matrix = None 

        # --- 1. 读取 extras (外参/内参) ---
        if os.path.exists(info["extras_path"]):
            try:
                df_extras = pd.read_parquet(info["extras_path"])
                if "episode_index" in df_extras.columns:
                    row = df_extras[df_extras["episode_index"] == episode_id]
                    if not row.empty:
                        # 验证并读取内参、外参
                        if "K_front" in row.columns:
                            k_front = np.array(row.iloc[0]["K_front"]).reshape(3, 3)
                            print(f"📷 [Extras] 成功加载 K_front 内参矩阵 (3x3).")
                            
                        if "Extrinsic_front" in row.columns:
                            ext_val = row.iloc[0]["Extrinsic_front"]
                            extrinsic_matrix = np.array(ext_val).reshape(4, 4)
                            print(f"🗺️ [Extras] 成功加载 Extrinsic_front 外参矩阵 (4x4).")
            except Exception as e:
                print(f"⚠️ [Extras] 读取 extras 发生错误: {e}")
        else:
            print(f"⚠️ [Extras] 未找到 {info['extras_path']}，将不应用坐标系转换。")

        # # --- 2. 读取轨迹数据 (data/chunk-xxx/...) ---
        # if os.path.exists(parquet_path):
        #     try:
        #         df = pd.read_parquet(parquet_path)
        #         # 适配新版标准 LeRobot 格式 (通常存放观测状态的 key 为 observation.state，或者你需要提取 action)
        #         # 这里假设你需要的是坐标点，字段名可根据实际情况修改
        #         target_col = 'action' if 'action' in df.columns else 'observation.state'
                
        #         if target_col in df.columns:
        #             raw_data = np.stack(df[target_col].to_numpy())
        #             gt_temp = []
                    
        #             for item in raw_data:
        #                 if item.size == 3:
        #                     gt_temp.append(item)
        #                 elif item.size == 16: 
        #                     mat = item.reshape(4, 4) if item.ndim == 1 else item
        #                     gt_temp.append(mat[:3, 3]) 
        #                 elif item.size >= 6:
        #                     gt_temp.append(item[:3])

        #             if len(gt_temp) > 0:
        #                 gt_trajectory = np.array(gt_temp)
        #                 print(f"📈 [Trajectory] 成功读取 {len(gt_trajectory)} 个轨迹点。")

        #                 # --- 外参转换 (相机 -> 机体) ---
        #                 if extrinsic_matrix is not None:
        #                     ones = np.ones((len(gt_trajectory), 1))
        #                     points_homo = np.hstack([gt_trajectory, ones]) 
        #                     transformed = (extrinsic_matrix @ points_homo.T).T 
        #                     gt_trajectory = transformed[:, :3]
        #                     print(f"🔄 [Transform] 已应用 Extrinsic_front (Camera -> Body) 坐标转换。")

        #                 # --- 归一化逻辑 ---
        #                 raw_start = gt_trajectory[0]
        #                 offset = np.array([0.0, 0.0, 1.5]) - raw_start
        #                 gt_trajectory = gt_trajectory + offset
                        
        #                 start_pos = gt_trajectory[0]
        #                 goal_pos = gt_trajectory[-1]
        #                 print(f"📍 [Position] 起点: {start_pos} | 终点: {goal_pos}")

        #     except Exception as e:
        #         print(f"❌ [Trajectory] 读取 Parquet 文件失败 {parquet_path}: {e}")
        # else:
        #     print(f"❌ [Trajectory] 找不到轨迹文件: {parquet_path}")
        # --- 2. 读取轨迹数据 (data/chunk-xxx/...) ---
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                target_col = 'action'
                
                if target_col in df.columns:
                    # 将 pandas 列转为 python 原生 list
                    raw_actions = df[target_col].tolist()
                    gt_temp = []
                    
                    for idx, item in enumerate(raw_actions):
                        try:
                            # 你的数据是标准的 4x4 矩阵 (列表嵌套列表)
                            # 直接精确提取前三行的第 4 个元素 (索引为 3)
                            raw_x = item[0][3]
                            raw_y = item[1][3]
                            raw_z = item[2][3]

                            x = raw_x
                            y = raw_y
                            z = raw_z
                            
                            gt_temp.append(np.array([x, y, z], dtype=np.float32))
                        except Exception as e:
                            print(f"⚠️ [Trajectory] 第 {idx} 帧解析失败, 数据内容: {item}, 报错: {e}")

                    if len(gt_temp) > 0:
                        gt_trajectory = np.array(gt_temp)
                        print(f"📈 [Trajectory] 成功从 {ep_str} 读取 {len(gt_trajectory)} 个真实的轨迹点！")

                        # --- 归一化逻辑 ---

                        start_pos = gt_trajectory[0]
                        goal_pos = gt_trajectory[-1]
                        print(f"📍 [Position] 真实起点: [{start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f}] | 真实终点: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")
                    else:
                        print(f"⚠️ [Trajectory] {ep_str} 提取后轨迹点为空！")

            except Exception as e:
                import traceback
                print(f"❌ [Trajectory] 解析报错 {parquet_path}: {e}")
                traceback.print_exc() # 打印详细报错信息，方便排查
        else:
            print(f"❌ [Trajectory] 找不到轨迹文件: {parquet_path}")

        print("="*50)

        safe_scene_id = info['scene_id'].replace('/', '_').replace('\\', '_')

        return {
            "episode_id": f"{safe_scene_id}_ep{episode_id}", 
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
    parser.add_argument("--scene", type=str, default=None, help="手动强制加载指定的场景 USD 文件名 (不带.usd后缀)")
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

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    global_metrics_file = os.path.join("results", f"metrics_run_{timestamp}.json")
    print(f"📝 本次运行的所有评估结果将追加保存至: {global_metrics_file}")

    # 4. 循环测试
    for i in range(len(dataset)):
        data = dataset.get_episode(i)
        if args.scene:
            scene_id = args.scene
            print(f"🔧 [Override] 已强制使用命令行指定的场景: {scene_id}")
        else:
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
        ep_metrics = runner.run_episode(data, map_size_info, max_steps=600, K=5, is_stop=1e-5)

        if ep_metrics is not None:
            with open(global_metrics_file, "a", encoding="utf-8") as f:
                # json.dumps 把它转成一行字符串，加 "\n" 保证每一局占一行
                f.write(json.dumps(ep_metrics) + "\n")

    print("All done!")
    simulation_app.close()

if __name__ == "__main__":
    main()