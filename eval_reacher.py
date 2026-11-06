import gymnasium as gym
from stable_baselines3 import PPO
import pandas as pd
import time

print("开始训练（只需几秒钟）...")
# 1. 创建训练环境
train_env = gym.make("Reacher-v5")
model = PPO("MlpPolicy", train_env, verbose=0)
model.learn(total_timesteps=10000) 
model.save("ppo_reacher_mini")
print("训练完成！准备打开 3D 窗口...")

# 2. 创建评估环境（开启可视化）
eval_env = gym.make("Reacher-v5", render_mode="human")

records = []
for ep in range(30):
    obs, info = eval_env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # 【关键修改】这里改成 0.1，让动作慢放，你就能看清机械臂了！
        time.sleep(0.1) 
    
    success = total_reward > -30
    records.append({"episode": ep, "total_reward": total_reward, "success": success})
    print(f"Episode {ep}: reward={total_reward:.2f}, success={success}")

# 3. 保存数据
df = pd.DataFrame(records)
df.to_csv("policy_eval_results.csv", index=False)
print(f"\n平均奖励: {df['total_reward'].mean():.2f}")
print(f"成功率: {df['success'].mean()*100:.1f}%")
print("测试完成！数据已保存。")

eval_env.close()