# eval_drone_core.py
# ==============================================================================
# 作用: 核心功能库 (Core Library)
# 职责:
#   1. DroneController: 底层姿态控制器 (PID)
#   2. BenchmarkRecorder: 可视化绘制与数据记录
#   3. DroneRunner: 单局任务执行流程管理
# 使用方法：本文件不需要在终端调用，由main.py文件调用
# ==============================================================================
import numpy as np
import json_numpy
import argparse
import os
import json
import cv2
import carb

from scipy.spatial.transform import Rotation as R

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim 
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects import VisualSphere
from pxr import UsdPhysics, PhysxSchema
from isaac_http_client import IsaacHTTPClient

# ==============================================================================
# [CONTROLLER] 无人机底层控制器
# ------------------------------------------------------------------------------
# 作用：将模型输出的 "位移/旋转指令" 转换为物理引擎的 "速度/力指令"。
# 原理：纯闭环 PD 控制
# ==============================================================================
class DroneController:
    def __init__(self, kp_pos=5.0, kd_pos=2.0, kp_rot=2.0, max_speed=2.0):
        """
        纯闭环 PD 控制器
        :param kp_pos: 位置 P 增益 (拉力：把无人机拉向期望位置)
        :param kd_pos: 位置 D 增益 (阻力：防止冲过头，抵抗当前速度)
        :param kp_rot: 旋转 P 增益
        :param max_speed: 最大速度限制
        """
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_rot = kp_rot
        self.max_speed = max_speed

    def compute_command(self, current_pos, current_vel, current_yaw, action_xyz, action_yaw_deg):
        """
        根据当前状态和模型的动作(期望位移)，计算控制指令
        :param action_xyz: 模型输出的 [dx, dy, dz] (世界坐标系下的期望增量)
        :param action_yaw_deg: 模型输出的 yaw 增量 (度)
        Input: 当前状态 + 模型给出的 Action (dx, dy, dz, dyaw)
        Output: 目标线速度 + 目标角速度
        """
        # 将局部位移指令转换回全局位移，以便输入给物理引擎
        cos_y = np.cos(current_yaw)
        sin_y = np.sin(current_yaw)
        
        global_dx = action_xyz[0] * cos_y - action_xyz[1] * sin_y
        global_dy = action_xyz[0] * sin_y + action_xyz[1] * cos_y
        global_dz = action_xyz[2]

        # 1. 期望位置 = 当前位置 + 模型输出的位移
        #    误差 Error = 期望位置 - 当前位置 = 模型输出的位移 (dx, dy, dz)
        pos_error = np.array([global_dx, global_dy, global_dz], dtype=np.float64)
        
        # 2. PD 控制公式: V_cmd = Kp * Error - Kd * V_current
        target_lin_vel = (self.kp_pos * pos_error) - (self.kd_pos * current_vel)
        
        # 3. 速度限幅
        speed = np.linalg.norm(target_lin_vel)
        if speed > self.max_speed:
            target_lin_vel = target_lin_vel / speed * self.max_speed
            
        # 4. 角度控制 (P控制)
        #    期望Yaw = 当前Yaw + 模型输出dYaw
        yaw_error_rad = np.deg2rad(action_yaw_deg)
        target_ang_vel_z = self.kp_rot * yaw_error_rad
        
        return target_lin_vel, np.array([0, 0, target_ang_vel_z])

# ==============================================================================
# [RECORDER] 核心可视化模块
# ------------------------------------------------------------------------------
# 作用：绘制 FPV + Local Map + Global Plot 三合一图，并保存视频。
# ==============================================================================
class BenchmarkRecorder:
    def __init__(self, start_pos, goal_pos, map_size=20.0, map_center_pos=None, gt_trajectory=None, target_waypoints=None, output_dir="results", goal_threshold=0.5):
        # goal_threshold是判决成功的距离，小于这个距离判定为成功
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.goal_threshold = goal_threshold

        self.frames = []
        self.trajectory = []
        self.collisions = 0
        self.min_dist_to_goal = float('inf')

        self.target_waypoints = target_waypoints

        # === [CONFIG] 画图板参数 ===
        self.map_world_size = map_size # 物理尺寸 (米)
        self.map_img_size = 640        # 像素尺寸 (px)

        # 确定画图板中心
        if map_center_pos is None:
            self.map_center = (self.start_pos + self.goal_pos) / 2
        else:
            self.map_center = map_center_pos
        self.map_center[2] = 0

        self.pixels_per_meter = self.map_img_size / self.map_world_size

        # 创建纯黑背景图 (Canvas) 用于画全局轨迹
        self.traj_canvas = np.zeros((self.map_img_size, self.map_img_size, 3), dtype=np.uint8)

        self.gt_trajectory = gt_trajectory

        # 预先画好静态元素 (GT, Waypoints)
        self._draw_static_elements()

    def world_to_pixel(self, pos):
        # 将世界坐标 (x, y) 转换为 地图图片像素坐标 (u, v)
        rel_x = pos[0] - self.map_center[0]
        rel_y = pos[1] - self.map_center[1]
        
        # 这里的映射关系 (u=-y, v=-x) 对应 Isaac Sim 的俯视相机朝向
        u = int(self.map_img_size / 2 - rel_y * self.pixels_per_meter)
        v = int(self.map_img_size / 2 - rel_x * self.pixels_per_meter)
        
        return (u, v)
    
    def _draw_static_elements(self):
        """在黑板上绘制不动的背景元素"""
        # 1. [TEST ONLY] Waypoints (青色小点) 这部分仅测试时使用，接模型后把这部分注释即可
        if self.target_waypoints is not None:
            for wp in self.target_waypoints:
                wp_px = self.world_to_pixel(wp)
                cv2.circle(self.traj_canvas, wp_px, 2, (255, 255, 0), -1)

        # 2. Ground Truth (红色线)
        if self.gt_trajectory is not None and len(self.gt_trajectory) > 1:
            gt_pts = [self.world_to_pixel(p) for p in self.gt_trajectory]
            cv2.polylines(self.traj_canvas, [np.array(gt_pts)], False, (0, 0, 255), 2)
        else:
            # 参考直线 (红色线)，当没有 GT 时给一个大致方向参考，起终点的连线
            start_px = self.world_to_pixel(self.start_pos)
            goal_px = self.world_to_pixel(self.goal_pos)
            cv2.line(self.traj_canvas, start_px, goal_px, (0, 0, 255), 2, cv2.LINE_AA)

        # 3. 终点标记 (红圈)
        goal_px = self.world_to_pixel(self.goal_pos)
        cv2.circle(self.traj_canvas, goal_px, 6, (0, 0, 255), 2)
        cv2.putText(self.traj_canvas, "Goal", (goal_px[0]+10, goal_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    def record_step(self, img_fpv, img_map, position, is_collision):
        """每帧调用：更新轨迹并拼图"""
        current_pos = np.array(position)
        self.trajectory.append(current_pos)
        # 1. 更新黑底轨迹图 (Global Plot)
        if len(self.trajectory) > 1:
            p1 = self.world_to_pixel(self.trajectory[-2])
            p2 = self.world_to_pixel(self.trajectory[-1])
            # 画绿色轨迹线
            cv2.line(self.traj_canvas, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
        
        # 拷贝一份画布用于显示当前位置 (避免箭头残留)
        canvas_display = self.traj_canvas.copy()
        curr_px = self.world_to_pixel(current_pos)
        cv2.circle(canvas_display, curr_px, 5, (0, 255, 0), -1)

        # 2. 图像拼接排版 (Layout)
        if img_fpv is not None and img_map is not None:
            # FPV (左图)
            fpv_rgb = img_fpv[:, :, :3][..., ::-1].astype(np.uint8)
            h_fpv, w_fpv, _ = fpv_rgb.shape # (360, 640)
            
            # 右侧两张小图 (Follow Map 和 Global Plot)
            h_small = h_fpv // 2 # 180
            w_small = h_small    # 让小图是正方形 (180x180)
            
            # 右上：Follow Map (实时俯视图)
            map_rgb = img_map[:, :, :3][..., ::-1].astype(np.uint8)
            map_small = cv2.resize(map_rgb, (w_small, h_small))
            cv2.rectangle(map_small, (0,0), (w_small-1, h_small-1), (100,100,100), 2)
            cv2.putText(map_small, "Follow Map (Local)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 右下：Global Plot (黑底轨迹图)
            plot_small = cv2.resize(canvas_display, (w_small, h_small))
            cv2.rectangle(plot_small, (0,0), (w_small-1, h_small-1), (100,100,100), 2)
            cv2.putText(plot_small, "Trajectory (Global)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 拼合
            right_col = np.vstack((map_small, plot_small))
            frame_combined = np.hstack((fpv_rgb, right_col))

            # UI 装饰 (在最终大图上写字)
            cv2.putText(frame_combined, "FPV Camera", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if is_collision:
                # 在 FPV 中央显示撞墙提示
                cv2.putText(frame_combined, "COLLISION!", (w_fpv//2 - 80, h_fpv//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            self.frames.append(frame_combined)
        
        # 3. 碰撞计数
        if is_collision:
            self.collisions += 1

        # 4. 更新最小距离 (用于 Oracle Success)
        dist = np.linalg.norm(current_pos - self.goal_pos)
        if dist < self.min_dist_to_goal:
            self.min_dist_to_goal = dist

    def save(self, episode_id=0):
        """任务结束：保存视频和数据"""
        # === 保存视频 ===
        if len(self.frames) > 0:
            height, width, _ = self.frames[0].shape
            video_path = os.path.join(self.output_dir, f"video_{episode_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height)) 
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"🎬 视频已保存: {video_path}")

        # === 计算核心指标 (Metrics) ===
        trajectory = np.array(self.trajectory)
        final_pos = trajectory[-1]
        # NE: Navigation Error (终点距离)
        ne = np.linalg.norm(final_pos - self.goal_pos)
        # SR: Success Rate (是否到达)
        is_success = bool(ne < self.goal_threshold)
        # OS: Oracle Success (过程中是否曾经到达)
        is_oracle_success = bool(self.min_dist_to_goal < self.goal_threshold)
        
        # Path Length: 实际飞行距离
        actual_path_len = 0.0
        if len(trajectory) > 1:
            actual_path_len = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1))
        # 最短路径 (Shortest Path) ，测试用的直线欧氏距离，真实数据用GT即可。这里已经兼容了
        shortest_path_dist = np.linalg.norm(self.goal_pos - self.start_pos)
        if self.gt_trajectory is not None and len(self.gt_trajectory) > 1:
            gt_len = np.sum(np.linalg.norm(self.gt_trajectory[1:] - self.gt_trajectory[:-1], axis=1))
            shortest_path_dist = float(gt_len)
        
        spl = 0.0
        if is_success:
            spl = shortest_path_dist / max(actual_path_len, shortest_path_dist)

        metrics = {
            "episode_id": episode_id,
            "success": float(is_success),             # SR
            "oracle_success": float(is_oracle_success), # OS
            "navigation_error": float(ne),     # NE
            "spl": float(spl),                 # SPL
            "path_length": float(actual_path_len),
            "shortest_path": float(shortest_path_dist),
            "collision_count": self.collisions,
            "total_steps": len(self.trajectory)
        }
        
        # json_path = os.path.join(self.output_dir, f"metrics_{episode_id}.json")
        # with open(json_path, "w") as f:
        #     json.dump(metrics, f, indent=4)
            
        print("-" * 40)
        print(f"📊 测试报告 (Episode {episode_id})")
        print(f"   Success (SR): {is_success} (Dist: {ne:.2f}m < {self.goal_threshold}m)")
        print(f"   Oracle Success (OS): {is_oracle_success}")
        print(f"   SPL: {spl:.4f}")
        print(f"   Nav Error (NE): {ne:.2f} m")
        print(f"   Collisions: {self.collisions}")
        print("-" * 40)

        return metrics

# ==============================================================================
# [RUNNER] 执行器
# ------------------------------------------------------------------------------
# 作用：管理无人机加载、相机配置、Loop 循环、Server 通信
# ==============================================================================
class DroneRunner:
    def __init__(self, world, server_url):
        self.world = world
        self.client = IsaacHTTPClient(url=server_url)
        self.drone_path = "/World/Quadcopter"
        self.camera_fpv = None
        self.camera_map = None
        self.drone = None

    def setup_drone(self, drone_usd_path, start_pos, goal_pos):
        """每局开始前调用：加载无人机、设置相机、配置刚体"""
        # ==========================================================
        # 🌟 采用 RTX 渲染设置 🌟
        # ==========================================================
        settings = carb.settings.get_settings()
        settings.set("/rtx/reflections/enabled", True)
        settings.set("/rtx/shadows/enabled", True)
        settings.set("/rtx/post/autoExposure/enabled", True) 
        settings.set("/rtx/post/histogram/enabled", True)
        settings.set("/rtx/post/histogram/whiteScale", 2.0) # 觉得暗可以调到 3.0 或 4.0
        settings.set("/app/renderer/waitIdle", True)
        settings.set("/rtx/hydra/enabled", True)
        # ==========================================================

        if not os.path.exists(drone_usd_path):
            print(f"❌ 致命错误: 找不到无人机文件: {drone_usd_path}", flush=True)
            return None, None
        # 1. 加载无人机 
        if not prim_utils.is_prim_path_valid(self.drone_path):
            prim_utils.create_prim(self.drone_path, "Xform", usd_path=drone_usd_path, position=start_pos)
            self.drone = RigidPrim(self.drone_path, name="drone_rigid")
            self.world.scene.add(self.drone)
        else:
            # 如果无人机还在，直接瞬移到起点
            if self.drone is not None:
                self.drone.set_world_pose(position=start_pos)
                self.drone.set_linear_velocity(np.zeros(3))
                self.drone.set_angular_velocity(np.zeros(3))

        # 2. 设置相机
        desired_path = f"{self.drone_path}/chassis/front_cam"
        # 检查 chassis 是否存在（作为父节点）
        chassis_path = f"{self.drone_path}/chassis"
        if prim_utils.is_prim_path_valid(chassis_path):
            # 如果 chassis 存在，我们就把相机建在 chassis/front_cam
            target_cam_path = desired_path
        else:
            print(f"⚠️ 警告: 找不到 chassis 节点，相机将直接挂载到无人机根节点。", flush=True)
            target_cam_path = f"{self.drone_path}/front_cam" # 挂到根目录下

        # 设置 FPV 相机 (输入给模型的图)
        if self.camera_fpv is None:
            self.camera_fpv = Camera(
                prim_path=target_cam_path,
                translation=np.array([0.0, 0.0, 0.0]),
                frequency=20, resolution=(640, 360),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            self.camera_fpv.initialize()
            self.camera_fpv.add_distance_to_image_plane_to_frame() 


        # 设置 Local Map 相机 (跟随无人机，用于右上角小图)
        # map_center = np.array([start_pos[0], start_pos[1], 12.0])
        map_center = np.array([start_pos[0], start_pos[1], start_pos[2] + 1.5])
        if self.camera_map is None:
            self.camera_map = Camera(
                prim_path="/World/MapCamera",
                position=map_center,
                frequency=20, resolution=(640, 640),
                orientation=np.array([0.7071, 0.0, 0.7071, 0.0])
            )
            self.camera_map.set_projection_type("orthographic")
            # ⚠️俯视图模糊等问题可以调下面这两个光圈，一般12/20/40这几个值比较合适
            self.camera_map.set_horizontal_aperture(20.0)
            self.camera_map.set_vertical_aperture(20.0)
            self.camera_map.initialize()
        else:
            # 如果是第二集，仅仅更新一下俯视相机的起始位置即可
            self.camera_map.set_world_pose(position=map_center)
        

        # 3. 计算全局画图板需要的尺寸 (给 Recorder 用的黑底图)
        mid_point = (start_pos + goal_pos) / 2
        span = np.linalg.norm(start_pos - goal_pos)
        global_map_size = max(span * 1.5, 10.0) #自适应，保证尺度不会超

        print(f"📷 [Setup] Global Map Size: {global_map_size:.1f}m")
        return global_map_size, mid_point

    def run_episode(self, episode_data, map_size_info, max_steps=2000, K=5, is_stop=1e-5, physics_dt=1.0/60.0):
        """执行单局任务的主循环"""
        start_pos = episode_data['start_pos']
        goal_pos = episode_data['goal_pos']
        instruction = episode_data['instruction']
        gt_traj = episode_data.get('gt_trajectory', None)
        episode_id = episode_data['episode_id']

        global_map_size, global_map_center = map_size_info

        # 1. 初始化刚体及物理状态重置
        self.world.reset()
                
        try:
            self.drone.enable_rigid_body_physics()
            self.drone.set_mass(1.0)
            # 禁用重力 (假设模型只输出位移，不控制推力)
            prim = prim_utils.get_prim_at_path(self.drone_path)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
        except: pass

        self.world.step(render=False)
        if self.drone is not None:
            self.drone.set_world_pose(position=start_pos)
            self.drone.set_linear_velocity(np.zeros(3))
            self.drone.set_angular_velocity(np.zeros(3))
            
            # 给物理引擎几帧时间来消化这个瞬移操作
            for _ in range(5):
                self.world.step(render=False)

        # 2. 预热物理引擎
        for _ in range(20): self.world.step(render=True)
        
        target_velocity = np.zeros(3)
        target_angular_vel = np.zeros(3)
        step_count = 0
        
        # #################################################################
        # [TEST ONLY] 临时测试代码：多路点折线飞行
        # 说明：正式接 Server 时，把下面这块定义 waypoints 的代码删掉！
        # #################################################################
        # 即使 Main 传进来一个目标，我们这里强行改写成一组折线，为了测试转弯
        # waypoints = []
        # total_steps = 32
        # vec_to_goal = goal_pos - start_pos
        # total_dist = np.linalg.norm(vec_to_goal)
        # if total_dist > 0.01:
        #     for i in range(total_steps):
        #         t = i / (total_steps - 1) # 确保最后一个点 t=1.0，完全重合
        #         # 线性插值基准线
        #         p = (1-t) * start_pos + t * goal_pos
        #         p[1] += 0.2 * np.sin(t * np.pi) 
        #         waypoints.append(p)
        # else:
        #     waypoints.append(goal_pos)


        # if gt_traj is not None and len(gt_traj) > 0:
        #     # 直接使用解析出来的真实轨迹作为路点
        #     waypoints = gt_traj
        # else:
        #     # 万一没读到轨迹，保底只飞终点
        #     waypoints = [goal_pos]
        # #################################################################

        # 初始化 Recorder
        recorder = BenchmarkRecorder(
            start_pos, 
            goal_pos, 
            map_size=global_map_size,        
            map_center_pos=global_map_center, 
            gt_trajectory=gt_traj, 
            target_waypoints=waypoints, 
            output_dir="results"
        )

        current_wp_idx = 0 # [TEST ONLY]这个是测试时waypoints的索引初始化，接模型可以注释掉
        
        # Kp拉力，Kd阻力，Kp_rot转向
        # ⚠️如果跑太慢或者转向速度什么的要调，可以在这里修改，也可以在server端修改衰减
        controller = DroneController(kp_pos=30.0, kd_pos=1.0, kp_rot=15.0, max_speed=2.0)

        print(f"🚀 Running Episode {episode_id} | Goal: {goal_pos}")

        # =========================================================================
        # 动作缓冲池初始化
        # =========================================================================
        # 这里的 K 是你想连续执行的步数
        K_EXEC_STEPS = K  
        # 停止判定的阈值 (位移小于 10微米)
        STOP_THRESHOLD = is_stop 
        
        # 动作队列：存放从Server拿回来的后续几步动作
        # 使用 deque 或 list 都可以，pop(0) 即可
        action_queue = [] 
        # 计数器：记录当前这批动作已经执行了多少步
        steps_since_inference = 0 
        # =========================================================================

        while step_count < max_steps:
            # 1. [PHYSICS] 执行控制指令
            self.drone.set_linear_velocity(target_velocity)
            self.drone.set_angular_velocity(target_angular_vel) 
            self.world.step(render=True)
            step_count += 1

            # 2. [SENSOR] 获取传感器数据
            rgb = self.camera_fpv.get_rgb()
            depth = self.camera_fpv.get_depth()
            pos, ori = self.drone.get_world_pose()
            current_vel = self.drone.get_linear_velocity()
            curr_rot = R.from_quat([ori[1], ori[2], ori[3], ori[0]])
            current_yaw = curr_rot.as_euler('zyx')[0]

            # [VISUALIZATION] 更新 Follow Camera 位置
            # new_cam_pos = np.array([pos[0], pos[1], 12.0])
            new_cam_pos = np.array([pos[0], pos[1], pos[2] + 1.5])
            self.camera_map.set_world_pose(position=new_cam_pos)
            
            # 3. [RECORD] 记录数据与画图
            actual_spd = np.linalg.norm(self.drone.get_linear_velocity())
            target_spd = np.linalg.norm(target_velocity)
            is_col = (target_spd > 0.1 and actual_spd < target_spd * 0.5)
            recorder.record_step(rgb, self.camera_map.get_rgb(), pos, is_col)

            # =============================================================
            # [STRATEGY] 策略层
            # =============================================================
            # [TEST ONLY] Mock 策略: 检查 Waypoints，真实接入模型请删除这一段
            # ⚠️接模型后，对应这部分逻辑，模型返回的action可以是追着往前几个点的，逻辑在server端改即可

            # 裁判逻辑：还是看当前路点 (check_point)
            # 我们必须到了 p[i]，才能算通过，才能切换下一个
            # check_target = waypoints[current_wp_idx]
            # dist_to_check = np.linalg.norm(check_target - pos)

            # if dist_to_check < 0.2: # 判定通过的阈值
            #     current_wp_idx += 1
            #     if current_wp_idx >= len(waypoints):
            #         current_wp_idx = len(waypoints) - 1
            
            # # 导航逻辑：给 Server 看前视点 (aim_point)
            # lookahead_t = 5  # [调节这个 t]：t 越大，看越远，飞得越快越平滑，但太大会切内圈
            
            # # 防止索引越界
            # aim_idx = min(current_wp_idx + lookahead_t, len(waypoints) - 1)
            # aim_target = waypoints[aim_idx] 
            # =============================================================
            # =============================================================

            # 计算指南针 (Compass/Yaw)
            rot = R.from_quat([ori[1], ori[2], ori[3], ori[0]])
            yaw_rad = rot.as_euler('zyx')[0] # 提取 Z 轴旋转 (弧度)

            # 计算 GPS (相对于起点的位移) 
            # ⚠️可能不需要，但是我看Habitat里面有，以防万一我就加上了，理论上这部分不该给模型看
            rel_gps = pos - start_pos

            policy_data = np.concatenate([pos, ori]) # [TEST ONLY]目前状态，接模型应该不给这个，仅测试用
            # [TEST ONLY] Obs 这个是加上起终点的，只是用来测试，真实模型不给这个，用下面那个
            # obs = {
            #     "rgb": rgb[:, :, :3] if rgb is not None else np.zeros((360, 640, 3)),
            #     "depth": depth if depth is not None else np.zeros((360, 640)),
            #     # "goal_pose": goal_pos,      #正常运行用的
            #     "goal_pose": aim_target,  #测试用的，加了几个waypoints的
            #     "policy": policy_data, 
            #     "instruction": instruction, # 传入指令
            #     "step": step_count,
            #     "collision": is_col
            # }

            # [MODEL INTERFACE] 构造 Observation真实模型obs，真实接模型用这个
            obs = {
                "rgb": rgb[:, :, :3] if rgb is not None else np.zeros((360, 640, 3)),
                "depth": depth if depth is not None else np.zeros((360, 640)),
                "instruction": instruction, # 核心输入：告诉模型去哪
                "step": step_count,         # 步数
                "compass": np.array([yaw_rad]), # 朝向 (Radians)
                "gps": rel_gps[:2],             # 相对位移 (只取x,y，通常不看高度z)
                "collision": is_col             # 碰撞检测
            }


            # =============================================================
            # 🔥 核心推理逻辑：Buffer 机制 🔥
            # =============================================================           
            # 初始化当前步的动作
            dx, dy, dz, yaw_deg = 0.0, 0.0, 0.0, 0.0
            
            # 判断是否需要请求 Server
            # 条件：Buffer 空了 OR 已经执行了 K 步
            need_inference = (len(action_queue) == 0) or (steps_since_inference >= K_EXEC_STEPS)

            if need_inference:
                # 1. 发送请求
                response = self.client.query(obs)
                # 2. 重置计数器
                steps_since_inference = 0
                action_queue = [] # 清空旧的（如果还有剩余的话，Receding Horizon通常丢弃剩余）

                result = json_numpy.loads(response)
                if isinstance(result, str):
                    result = json_numpy.loads(result)

                if result and "action" in result:
                    # 获取 [N, 4] 的数组
                    new_actions = result["action"] # 应该是一个 list of lists 或者 numpy array
                    
                    # 这里的 new_actions 如果是 list，可以直接用
                    # 如果是 numpy，确保转为可迭代对象
                    if hasattr(new_actions, 'tolist'):
                        new_actions = new_actions.tolist()
                    
                    action_queue.extend(new_actions)

            # --- 无论是否刚请求过，都从 Buffer 里取出一个动作执行 ---
            if len(action_queue) > 0:
                # 取出队首动作
                current_action = action_queue.pop(0) 
                dx, dy, dz, yaw_deg = current_action
                steps_since_inference += 1
            else:
                # 万一 Server 挂了或者没返回东西，保持静止
                dx, dy, dz, yaw_deg = 0, 0, 0, 0

            # 自动 Stop 判定
            server_stop = False
            # 检查绝对值是否足够小
            if (abs(dx) < STOP_THRESHOLD and 
                abs(dy) < STOP_THRESHOLD and 
                abs(dz) < STOP_THRESHOLD):
                server_stop = True
            
            # 碰撞保护：撞墙后停止动作
            if is_col:
                print("💥 撞墙了！强制刹车！")
                dx, dy, dz = 0, 0, 0
                yaw_deg = 0 
                # ⚠️如果需要也可以在这里给一个反向速度让它弹回来
                # dx = -target_velocity[0] * 0.5 * physics_dt

            # 5. [CONTROL] 计算物理指令
            target_velocity, target_angular_vel = controller.compute_command(
                current_pos=pos,
                current_vel=current_vel,
                current_yaw=current_yaw,
                action_xyz=[dx, dy, dz],
                action_yaw_deg=yaw_deg
            )
            # print(f"target_velocity：{target_velocity}，target_angular_vel：{target_angular_vel}")
            # linear_speed = np.linalg.norm(target_velocity)
            # print(f"线速度大小: {linear_speed:.2f} m/s | 速度向量: {target_velocity} | 角速度: {target_angular_vel}")

            # 6. [STOP] 停止条件
            
            # [TEST ONLY]这个是测试的时候用的！因为加了waypoints不让他中间就停下
            # if server_stop:
            #     # 情况 1：只是到了中间的一个路点 -> 切下一个，继续飞
            #     if current_wp_idx < len(waypoints) - 1:
            #         # print(f"✅ Waypoint {current_wp_idx} Reached. Next!") # 调试用
            #         current_wp_idx += 1
            #         current_target = waypoints[current_wp_idx]
            #         # 注意：这里不要 break，让循环继续，去追下一个点
                    
            #     # 情况 2：确实是最后一个点 -> 任务结束
            #     else:
            #         print("🎉 Mission Complete (All Waypoints Covered).")
            #         # 悬停一会儿再退
            #         for _ in range(20): 
            #             self.drone.set_linear_velocity(np.zeros(3))
            #             self.drone.set_angular_velocity(np.zeros(3))
            #             self.world.step(render=True)
            #         break

            # [MODEL]模型预测的时候用下面这个就行
            if server_stop:
                print("✅ Server requested STOP. Mission Complete.")
                # 悬停展示一会儿
                for _ in range(20): 
                    # 给一个 0 速度让它稳住
                    self.drone.set_linear_velocity(np.zeros(3))
                    self.drone.set_angular_velocity(np.zeros(3))
                    self.world.step(render=True)
                break

        ep_metrics = recorder.save(episode_id)
        print(f"🏁 Episode {episode_id} Done.")
        return ep_metrics