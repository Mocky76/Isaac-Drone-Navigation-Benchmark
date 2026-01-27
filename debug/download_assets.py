# download_assets.py
from isaacsim import SimulationApp

# 启动仿真器 (必须)
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
import os

# === 配置 ===
# 1. 目标：官方在线仓库的地址
ONLINE_URL = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2022.2.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"

# 2. 存到哪里：你的本地 assets 目录
LOCAL_PATH = "/isaac-sim/our_benchmark/assets/scenes/warehouse.usd"

def main():
    print(f"1. 正在尝试连接在线资源: {ONLINE_URL}")
    print("   (这可能需要几分钟，取决于你的网速...)")
    
    # 打开在线舞台
    try:
        stage = Usd.Stage.Open(ONLINE_URL)
    except Exception as e:
        print(f"Error: 无法连接网络资源。请检查容器是否能联网。\n{e}")
        return

    if not stage:
        print("Error: 场景加载失败！")
        return

    print("2. 资源已加载。正在打包(Flatten)并保存到本地...")
    print(f"   目标路径: {LOCAL_PATH}")

    # 确保文件夹存在
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

    # === 关键步骤：Flatten ===
    # 这会将所有引用的层级合并到一个文件中，解决材质丢失问题
    flattened_stage = stage.Flatten()
    
    # 保存到本地
    flattened_stage.Export(LOCAL_PATH)
    
    print("-" * 50)
    print("✅ 成功！")
    print(f"文件已保存至: {LOCAL_PATH}")
    print("你现在可以在 eval_drone_core.py 中使用这个本地路径了！")
    print("-" * 50)

    simulation_app.close()

if __name__ == "__main__":
    main()