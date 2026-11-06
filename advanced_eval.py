import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque

if not os.path.exists("ppo_reacher_mini.zip"):
    print("❌ 找不到模型文件！请先运行一次 eval_reacher.py 训练并保存模型。")
    exit()

model = PPO.load("ppo_reacher_mini")

def evaluate(is_perturbed=False):
    eval_env = gym.make("Reacher-v5", render_mode=None) 
    total_rewards_list = []
    total_episodes = 20
    
    for ep in range(total_episodes):
        obs, info = eval_env.reset()
        
        if is_perturbed:
            try:
                mujoco_model = eval_env.get_wrapper_attr('model')
                mujoco_model.body_mass[:] *= 2.5        # 质量 2.5 倍
                mujoco_model.dof_frictionloss[:] = 0.5  # 适度摩擦
            except Exception as e:
                pass
            
        total_reward = 0
        done = False
        
        # 延迟 2 步
        action_queue = deque([np.zeros(eval_env.action_space.shape, dtype=np.float32)] * 2, maxlen=2)
        
        while not done:
            if is_perturbed:
                obs_array = np.array(obs, dtype=np.float32)
                noisy_obs = obs_array + np.random.normal(0, 0.1, size=obs_array.shape)
                if np.isnan(noisy_obs).any():
                    noisy_obs = np.nan_to_num(noisy_obs, nan=0.0)
            else:
                noisy_obs = obs
                
            action, _ = model.predict(noisy_obs, deterministic=True)
            
            if is_perturbed:
                # 动作加入稍微温和一点的噪声 (0.1)
                action = action + np.random.normal(0, 0.1, size=action.shape)
                actual_action = action_queue[0]
                action_queue.append(action)
            else:
                actual_action = action
                
            obs, reward, terminated, truncated, info = eval_env.step(actual_action)
            
            # 【关键修复】如果物理引擎崩溃算出 NaN，直接记极差惩罚并强制结束本回合
            if np.isnan(reward):
                reward = -50
                terminated = True
                # print("⚠️ 物理引擎崩溃!")
                
            total_reward += reward
            done = terminated or truncated
            
        total_rewards_list.append(total_reward)
            
    eval_env.close()
    return np.mean(total_rewards_list)

print("1. 正在测试标准环境...")
normal_reward = evaluate(is_perturbed=False)

print("2. 正在测试重度扰动环境 (Sim2Real 挑战)...")
perturbed_reward = evaluate(is_perturbed=True)

print(f"\n✅ 标准环境平均奖励: {normal_reward:.2f}")
print(f"❌ 重度扰动环境平均奖励: {perturbed_reward:.2f}")

print("\n3. 正在生成对比图表...")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

categories = ['标准环境', 'Sim2Real 扰动环境']
rewards = [normal_reward, perturbed_reward]

plt.figure(figsize=(6, 5))
colors = ['#4CAF50', '#F44336']
bars = plt.bar(categories, rewards, color=colors, width=0.5)

plt.ylabel('平均奖励 (越接近 0 越好)')
plt.title('策略在 Sim2Real 物理扰动下的性能衰减')

# 安全设置 Y 轴，确保图表能看到明显的落差
plt.ylim(min(rewards) - 10, 2) 

for bar in bars:
    yval = bar.get_height()
    text_y = yval - 2 if yval < 0 else yval + 1
    va_align = 'top' if yval < 0 else 'bottom'
    plt.text(bar.get_x() + bar.get_width()/2, text_y, f"{yval:.1f}", ha='center', va=va_align, fontweight='bold', fontsize=12)

plt.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存为 robustness_comparison.png")