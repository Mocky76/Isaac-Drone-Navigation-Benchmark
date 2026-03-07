# drone_server.py
# ==============================================================================
# 作用: 无人机控制策略服务器 (Policy Server)
# 职责:
#   1. 接收 Client 发来的观测数据 (Observation)
#   2. 运行策略 (Mock算法 或 深度学习模型)
#   3. 返回控制指令 (Action)
# 使用方法：: /isaac-sim/python.sh drone_server.py
# ==============================================================================
from fastapi import FastAPI, Request
import json_numpy
import numpy as np
import time
import uvicorn
import torch
from scipy.spatial.transform import Rotation as R

# 开启 numpy 序列化支持
json_numpy.patch()

app = FastAPI()

# [MODEL INIT] 在这里加载模型
# print("⏳ Loading Deep Learning Model...")
# model = VLABaseModel.from_pretrained("path/to/checkpoint")
# model.eval()
# model.to("cuda")
# print("✅ Model Loaded!")

@app.post("/reset")
def reset():
    """
    [LIFECYCLE] 重置接口
    作用：当一个新的 Episode 开始时调用，在这里清空 Hidden State 或 History Buffer这类记忆
    """

    return jsonify({"status": "ok"})

@app.post("/act")
def act(req: dict):
    """
    核心推理接口
    Input (JSON):
        {
            "observation": {
                "rgb": [H, W, 3],       # 视觉输入
                "depth": [H, W],        # 深度输入 (可选)
                "instruction": "text",  # 文本指令
                "gps": [x, y],          # 相对位移 (可选)
                "compass": [yaw],       # 相对朝向 (可选)
            }
        }
    
    Output (JSON):
        {
            "action": [dx, dy, dz, dyaw], # 局部坐标系下的位移和旋转增量
            "stop": bool                  # 是否到达终点
        }
    """
    # 初始化返回值
    dx, dy, dz, yaw_deg = 0.0, 0.0, 0.0, 0.0
    STOP = False

    # 解析数据
    # req = json_numpy.loads(request.data)
    obs = req["observation"]

    # #################################################################
    # [TEST ONLY]以下部分仅为测试用
    # #################################################################
    
    # 获取关键数据
    rgb_img = obs["rgb"]  
    depth = obs.get("depth")

    # 假设预测步长 (Chunk Size)，截图里看起来像是 16
    N = 16 
    
    # 初始化一个 [N, 4] 的 numpy 数组 (float32)
    # 格式: [dx, dy, dz, dyaw]
    pred_actions = np.zeros((N, 4), dtype=np.float32)

    # --- 获取测试用的导航目标 (仅用于生成假数据，实际接模型时不需要这部分) ---
    goal_pos = obs.get("goal_pose") 
    policy = obs.get("policy")
    
    if goal_pos is not None and policy is not None:
        current_pos = policy[:3]
        dist_to_goal = np.linalg.norm(goal_pos - current_pos)

        # 简单的逻辑：如果还没到终点，就填充动作
        if dist_to_goal > 0.02:
            import math
            # 这里为了演示，我们计算第一步的动作，然后简单的复制给后面几步
            # 在真实的 Diffusion Policy 中，每一步的动作通常是变化的（形成曲线）
            
            raw_dx = (goal_pos[0] - current_pos[0])
            raw_dy = (goal_pos[1] - current_pos[1])
            
            # 缩放系数
            scale = 0.2
            dx = np.clip(raw_dx * scale, -0.3, 0.3)
            dy = np.clip(raw_dy * scale, -0.3, 0.3)

            # 1. 计算目标方向的绝对偏航角 (弧度)
            target_yaw = math.atan2(raw_dy, raw_dx)
            
            # 2. 获取当前机身朝向 (从 Isaac Sim 传过来的四元数解析)
            ori = policy[3:7] # policy里面包含了位置和四元数 [qw, qx, qy, qz]
            rot = R.from_quat([ori[1], ori[2], ori[3], ori[0]]) # 转成 scipy 格式 [x,y,z,w]
            current_yaw = rot.as_euler('zyx')[0]
            
            # 3. 计算最短的旋转角度差 (将其限制在 -pi 到 pi 之间)
            dyaw_rad = target_yaw - current_yaw
            dyaw_rad = (dyaw_rad + np.pi) % (2 * np.pi) - np.pi
            
            # 4. 转成度数，并做个限幅防止转得太鬼畜 (每次最多转 5 度)
            dyaw_deg = np.degrees(dyaw_rad)
            dyaw_deg = np.clip(dyaw_deg, -5.0, 5.0)

            # 将全局偏差 (raw_dx, raw_dy) 转换到无人机局部坐标系
            cos_y = np.cos(current_yaw)
            sin_y = np.sin(current_yaw)
            
            # 全局转局部 (逆旋转)
            local_dx = raw_dx * cos_y + raw_dy * sin_y
            local_dy = -raw_dx * sin_y + raw_dy * cos_y
            
            # # 使用局部坐标系下的 dx, dy
            # dx = np.clip(local_dx * scale, -0.3, 0.3)
            # dy = np.clip(local_dy * scale, -0.3, 0.3)
            # 基于局部坐标系的航向控制策略
            # 策略1：算出机头对准系数 (如果偏角大于 45 度，该系数为 0)
            align_factor = max(0.0, 1.0 - abs(dyaw_rad) / (np.pi / 4))
            
            # 策略2：禁止倒车，只取前向距离
            forward_dist = max(0.0, local_dx)
            
            # 策略3：计算最终动作
            # 只有当机头大致对准目标时，才允许向前飞 (dx)，否则先原地转头
            dx = np.clip(forward_dist * scale * align_factor, 0.0, 0.3)
            
            # 强制干掉侧飞 (dy)，逼迫无人机像汽车一样只能往前开和转弯
            dy = 0.0
            
            # 填充数组
            # 示例：让未来 N 步都执行相同的向前动作（匀速直线运动）
            # 真实模型推理出来的 action 会自带时序变化
            pred_actions[:, 0] = dx  # dx
            pred_actions[:, 1] = dy  # dy
            pred_actions[:, 2] = 0.0 # dz
            pred_actions[:, 3] = dyaw_deg # dyaw (简化)

            # 为了让数据看起来更像截图里的真实数据（有微小噪声/变化），加一点随机抖动
            noise = np.random.normal(0, 0.001, (N, 4))
            pred_actions += noise
        else:
            # 到达终点，保持全 0，或者根据你的逻辑输出
            pass


    # #################################################################
    # #################################################################

    # #################################################################
    # 接真实模型逻辑大致如下：
    # #################################################################
    # 1. 提取输入
    # rgb_img = obs["rgb"]       
    # instruction = obs["instruction"] 
    # depth = obs["depth"]
    # gps = obs["gps"]         # 可选
    # compass = obs["compass"] # 可选

    # 2. 预处理
    # 例如：归一化、转 Tensor、Resize
    # tensor_img = torch.from_numpy(rgb_img).permute(2,0,1).float() / 255.0
    # inputs = tokenizer(instruction, return_tensors="pt")

    # 3. 模型推理
    # with torch.no_grad():
    #     pred_action = model(tensor_img, inputs)
    # pred_actions = pred_action.cpu().numpy()[0]
    # #################################################################

    # 返回结果
    response = {
        "action": pred_actions
    }
    
    return json_numpy.dumps(response)

if __name__ == "__main__":
    # 启动 Flask 服务，监听 9000 端口
    print(f"🚀 Drone Policy Server running on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)