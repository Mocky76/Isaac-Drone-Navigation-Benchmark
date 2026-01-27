# inspect_data.py
import pandas as pd
import json
import os
import numpy as np

# 指向你刚刚链接好的具体路径
base_path = "/home/sjy/workspace/our_benchmark/data/hm3d_zed/00001-UVdNNRcVyV1/trajectory_19"

def inspect_parquet():
    path = os.path.join(base_path, "data/chunk-000/episode_000000.parquet")
    print(f"\n🔍 正在检查 Parquet: {path}")
    
    if not os.path.exists(path):
        print("❌ 文件不存在！请检查路径链接是否成功。")
        return

    try:
        df = pd.read_parquet(path)
        print(f"✅ 加载成功！数据形状: {df.shape}")
        print("-" * 30)
        print("列名 (Columns):")
        print(df.columns.tolist())
        print("-" * 30)
        print("第一行数据 (Start?):")
        print(df.iloc[0])
        print("-" * 30)
    except Exception as e:
        print(f"❌ 读取失败: {e}")

def inspect_jsonl(filename):
    path = os.path.join(base_path, "meta", filename)
    print(f"\n🔍 正在检查 JSONL: {path}")
    
    if not os.path.exists(path):
        print(f"⚠️ {filename} 不存在，跳过。")
        return

    try:
        with open(path, 'r') as f:
            # 只读第一行
            line = f.readline()
            data = json.loads(line)
            print(f"✅ 第一条数据 Keys: {list(data.keys())}")
            print("内容预览:")
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    inspect_parquet()
    inspect_jsonl("episodes.jsonl")
    inspect_jsonl("tasks.jsonl")
    inspect_jsonl("info.json")