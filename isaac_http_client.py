import requests
import json_numpy
import numpy as np
import torch

json_numpy.patch()

class IsaacHTTPClient:
    def __init__(self, url):
        self.url = url

    def reset(self, env_id: int, **kwargs):
        """
        通知 Server 重置特定环境的记忆（如果 Server 是有状态的）
        """
        payload = {"type": "reset", "env_id": env_id, **kwargs}
        try:
            requests.post(self.url, data=json_numpy.dumps(payload), headers={"Content-Type": "application/json"}, timeout=1.0)
        except Exception as e:
            print(f"[Client] Reset warning: {e}")

    def query(self, obs: dict) -> dict:
        """
        发送观测数据，接收动作/轨迹
        """
        # 预处理：将 Tensor 转为 Numpy，确保 JSON 可序列化
        serializable_obs = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                serializable_obs[k] = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                serializable_obs[k] = v
            else:
                serializable_obs[k] = v

        payload = {"observation": serializable_obs}
        
        try:
            resp = requests.post(
                self.url,
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
                # timeout=5.0
            )
            resp.raise_for_status()
            return json_numpy.loads(resp.text)
        except requests.exceptions.RequestException as e:
            print(f"[Client] Query failed: {e}")
            return None