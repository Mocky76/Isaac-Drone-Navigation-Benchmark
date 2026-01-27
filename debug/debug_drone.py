# debug_drone.py
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from pxr import Usd, UsdPhysics, UsdGeom
import numpy as np
import os

def main():
    world = World(stage_units_in_meters=1.0)
    
    # 路径
    drone_usd = "/isaac-sim/our_benchmark/assets/robots/quadcopter.usd"
    
    if not os.path.exists(drone_usd):
        print(f"!!! 错误: 找不到文件 {drone_usd}")
        simulation_app.close()
        return

    print(f"正在读取模型文件: {drone_usd}")
    
    # 1. 创建模型
    drone_path = "/World/Quadcopter"
    create_prim(
        prim_path=drone_path,
        prim_type="Xform",
        usd_path=drone_usd,
        position=np.array([0, 0, 1.0])
    )

    # 2. 这里的关键是：我们不 step 物理引擎，直接读 USD 数据
    stage = world.stage
    
    print("-" * 30)
    print(" >>> 静态质量分析报告 <<< ")
    
    total_mass = 0.0
    found_mass_prim = False
    
    # 遍历所有 Prim，把所有定义的质量加起来
    for prim in stage.Traverse():
        # 检查是否有 MassAPI (质量属性)
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            if mass_attr.IsValid():
                m = mass_attr.Get()
                if m is not None and m > 0:
                    print(f" - 发现部件: {prim.GetPath()} | 质量: {m:.4f} kg")
                    total_mass += m
                    found_mass_prim = True
    
    # 如果没找到 MassAPI，尝试找 Collision API (有时候质量是根据体积算的)
    if not found_mass_prim:
        print("警告: 未直接找到 Mass 属性，尝试估算...")
        # 这里给一个常见无人机的默认值，防止卡死
        total_mass = 1.0 
    
    if total_mass < 0.001:
        print("警告: 读到的质量几乎为 0，模型可能没有配置物理属性！")
        total_mass = 1.0 # 强制兜底

    print("-" * 30)
    print(f"【最终结论】 无人机总质量: {total_mass:.4f} kg")
    
    gravity = 9.81
    hover_force = total_mass * gravity
    print(f"理论悬停总升力: {hover_force:.4f} N")
    print(f"单个电机悬停升力 (除以4): {hover_force/4:.4f} N")
    print("-" * 30)
    
    print("诊断完成，正在关闭...")
    simulation_app.close()

if __name__ == "__main__":
    main()