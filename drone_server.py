# drone_server.py
# ==============================================================================
# 作用: 无人机控制策略服务器 (Policy Server)
# 职责:
#   1. 接收 Client 发来的观测数据 (Observation)
#   2. 运行策略 (Mock算法 或 深度学习模型)
#   3. 返回控制指令 (Action)
# 使用方法：: python drone_server.py
# ==============================================================================
from flask import Flask, request, jsonify
import json_numpy
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# 开启 numpy 序列化支持
json_numpy.patch()

app = Flask(__name__)

# [MODEL INIT] 在这里加载模型
# print("⏳ Loading Deep Learning Model...")
# model = VLABaseModel.from_pretrained("path/to/checkpoint")
# model.eval()
# model.to("cuda")
# print("✅ Model Loaded!")

@app.route("/reset", methods=["POST"])
def reset():
    """
    [LIFECYCLE] 重置接口
    作用：当一个新的 Episode 开始时调用，在这里清空 Hidden State 或 History Buffer这类记忆
    """

    return jsonify({"status": "ok"})

@app.route("/act", methods=["POST"])
def act():
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
    req = json_numpy.loads(request.data)
    obs = req["observation"]

    # #################################################################
    # [TEST ONLY]以下部分仅为测试用
    # #################################################################
    
    # 获取关键数据
    rgb_img = obs["rgb"]  
    depth = obs.get("depth")
    
    goal_pos = obs.get("goal_pose") # [3]
    policy = obs.get("policy")      # [7] -> [x, y, z, qw, qx, qy, qz]
    
    # 2. 运行导航逻辑
    if goal_pos is None or policy is None:
        return json_numpy.dumps({"action": [0,0,0,0]})

    current_pos = policy[:3]
    current_ori = policy[3:] # [w, x, y, z]
    
    dist_to_goal = np.linalg.norm(goal_pos - current_pos)

    if dist_to_goal > 0.02:
        direction = (goal_pos - current_pos) / dist_to_goal
        # A. 计算位移 (3D)
        raw_dx = (goal_pos[0] - current_pos[0]) 
        raw_dy = (goal_pos[1] - current_pos[1])
        raw_dz = (goal_pos[2] - current_pos[2])
        
        # 限制单步最大输出 (比如最大只允许输出 0.2m 的位移)，防止飞飞了
        # 这种写法保留了方向和距离的比例关系
        scale_factor = 0.2 # 调节这个！越大飞得越快
        
        # 简单的线性缩放：动作 = 距离向量 * 系数
        # 如果距离很远，动作就会很大
        dx = np.clip(raw_dx * scale_factor, -0.3, 0.3)
        dy = np.clip(raw_dy * scale_factor, -0.3, 0.3)
        dz = np.clip(raw_dz * scale_factor, -0.3, 0.3)
        
        # B. 计算转弯 (Yaw)
        if dist_to_goal > 0.01:
            target_yaw = np.arctan2(direction[1], direction[0])
            
            # 转换四元数: Isaac [w,x,y,z] -> Scipy [x,y,z,w]
            q_scipy = [current_ori[1], current_ori[2], current_ori[3], current_ori[0]]
            curr_yaw = R.from_quat(q_scipy).as_euler('zyx')[0]
            
            diff = target_yaw - curr_yaw
            # 处理角度跳变
            while diff > np.pi: diff -= 2*np.pi
            while diff < -np.pi: diff += 2*np.pi
            
            # 限制转速
            raw_yaw_deg = np.clip(np.rad2deg(diff), -1.5, 1.5)
        else:
            raw_yaw_deg = 0.0

        if dist_to_goal > 0.5:
            decay = 1.0
        elif dist_to_goal < 0.1:
            decay = 0.0
        else:
            # 在 0.1 ~ 0.5m 之间，系数从 0 慢慢涨到 1
            decay = (dist_to_goal - 0.1) / 0.4

        final_yaw = np.clip(raw_yaw_deg, -10.0, 10.0) * decay
        
        # 打印日志方便看 Server 有没有在工作
        print(f"[Server] Dist: {dist_to_goal:.2f}m | Action: [{dx:.2f}, {dy:.2f}, {dz:.2f}, {yaw_deg:.1f}]")
    else:
        print("[Server] Target Reached (Dist < 0.02m). Sending STOP signal.")
        STOP = True
        dx, dy, dz, final_yaw = 0.0, 0.0, 0.0, 0.0

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
    #     pred_action, pred_stop = model(tensor_img, inputs)
        
    # 4. 解析输出 (Post-processing)
    # 假设模型输出是 [dx, dy, dz, dyaw]
    # dx, dy, dz, yaw_deg = pred_action.cpu().numpy()[0]
    # STOP = pred_stop.item() > 0.5

    # 5. [可选] 安全限制 (Safety Guard)
    # 如果生成的dx,dy,dz,yaw过大，可以在这里限制
    # dx = np.clip(dx, -0.5, 0.5)
    # final_yaw = np.clip(yaw_deg, -10.0, 10.0)
    # #################################################################

    # 返回结果
    response = {
        "action": [dx, dy, dz, final_yaw],
        "stop": STOP
    }
    
    return json_numpy.dumps(response)

if __name__ == "__main__":
    # 启动 Flask 服务，监听 9009 端口
    print(f"🚀 Drone Policy Server running on port 9009...")
    app.run(host="0.0.0.0", port=9009, debug=False)