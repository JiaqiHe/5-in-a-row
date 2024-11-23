#!/usr/bin/python
import argparse
from collections import deque
from ppo_agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--env_path",
    default=None,
    type=str,
    help="The Godot binary to use, do not include for in editor training",
)
parser.add_argument(
    "--viz",
    action="store_true",
    help="If set, the simulation will be displayed in a window during training. Otherwise "
    "training will run without rendering the simulation. This setting does not apply to in-editor training.",
    default=False,
)
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
parser.add_argument("--speedup", default=1, type=int, help="Whether to speed up the physics in the env")
parser.add_argument(
    "--n_parallel",
    default=1,
    type=int,
    help="How many instances of the environment executable to " "launch - requires --env_path to be set if > 1.",
)
args, extras = parser.parse_known_args()


# start Godot environment
env = StableBaselinesGodotEnv(
    env_path=args.env_path, show_window=args.viz, seed=args.seed, n_parallel=args.n_parallel, speedup=args.speedup
)
env = VecMonitor(env)

state_size = 204
action_size = 100
print_every = 5000


def main():
    print("Start Training...")
    agent = Agent(state_size, action_size, load_pretrained=False)
    scores = run_ppo(env, agent)
    print("\nTraining finished.")
    close_env()

    scores = np.array(scores)
    x = np.where(scores >= 1e5)
    print('Max score reached: {:.4f}'.format(np.amax(scores)))

    df = pd.DataFrame({
        'x': np.arange(len(scores)),
        'y': scores, 
        })
    rolling_mean = df.y.rolling(window=50).mean()

    img_path ="scores_plot.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(df.x, df.y, label='Scores')
    plt.plot(df.x, rolling_mean, label='Moving avg', color='orange')
    plt.ylabel('Scores')
    plt.xlabel('Episodes')
    plt.legend()
    fig.savefig(fname=img_path)
    print('\nPlot saved to {}.'.format(img_path))
    agent.save()


def run_ppo(env, agent, num_episodes=5_000_000, eval_freq=10):
    scores = []
    scores_window = deque(maxlen=100)
    
    obs = env.reset()  # 只在开始时重置一次
    
    for i_episode in range(1, num_episodes+1):
        agent.step(env)
        
        # 定期评估
        if i_episode % eval_freq == 0:
            # 创建评估环境的副本，避免影响训练环境
            env.reset()
            eval_env = env
            scores_as_two_players = agent.act(eval_env)
            score = np.array(scores_as_two_players).max()
            scores.append(score)
            scores_window.append(score)
            
            if i_episode % print_every == 0:
                print('\rEpisode {}/{}, Score: {:.4f}, Avg Score: {:.4f}'.format(
                    i_episode, num_episodes, score, np.mean(scores_window)))

    return scores

def close_env():
    try:
        print("closing env")
        env.close()
    except Exception as e:
        print("Exception while closing env: ", e)

if __name__ == "__main__":
    main()