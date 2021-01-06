from DoomEnv import DoomEnv, ENVS
import tensorflow as tf
from os import path
import numpy as np
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.sac.sac import SAC
from stable_baselines.her.her import HER
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from stable_baselines.common.noise import NormalActionNoise
from policy import PPOPolicy, SACPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

import argparse

parser = argparse.ArgumentParser(description="vizdoom")
parser.add_argument("-alg", type=str, default="sac", help="algorithm for training")
parser.add_argument("-env", type=int, default=0, help="which stage it is")
args = parser.parse_args()


class PPOTrain:
    def __init__(self, env_index: int = 0, max_time_step_per_robot=1000, model_path="./cnn", feature_extractor="cnn"):
        self.env = make_vec_env(DoomEnv, n_envs=2, vec_env_cls=DummyVecEnv,
                                env_kwargs={"display": False,
                                            "feature": feature_extractor,
                                            "env_index": env_index})

        policy_kwargs = dict(feature_extraction=feature_extractor)
        self.model = PPO2(PPOPolicy, self.env, policy_kwargs=policy_kwargs, gamma=0.99, n_steps=max_time_step_per_robot,
                          ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4,
                          noptepochs=100, cliprange=0.2, cliprange_vf=None, verbose=0, tensorboard_log="./tensorboard",
                          _init_setup_model=True, full_tensorboard_log=True, seed=None, n_cpu_tf_sess=None)

        self.model_path = model_path
        if path.exists(self.model_path):
            self.model.load(self.model_path, env=self.env)

    def train(self, total_time_step=50000):
        path = self.model_path
        self.model.learn(total_timesteps=total_time_step)
        self.model.save(path)

    def load(self, path=None):
        if path is None:
            path = self.model_path
        model = PPO2.load(path, env=self.env)


class SACTrain:
    def __init__(self, env_index: int = 0, model_path="./", feature_extractor="cnn"):
        self.env = DoomEnv(display=False, feature=feature_extractor, env_index=env_index, learning_type="sac")
        self.model = SAC(SACPolicy, self.env, verbose=1, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                         learning_starts=100, train_freq=1, batch_size=256,
                         tau=0.005, ent_coef='auto', target_update_interval=1,
                         gradient_steps=1, target_entropy='auto', action_noise=None,
                         random_exploration=1, tensorboard_log="./tensorboard",
                         _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                         seed=None, n_cpu_tf_sess=None)

        self.model_path = model_path
        if path.exists(self.model_path):
            self.model.load(self.model_path, env=self.env)

    def train(self, total_time_step=5e7, saving_path="./cnn"):
        self.model.learn(total_timesteps=total_time_step, log_interval=10)
        self.model.save(saving_path)

    def load(self, loading_path=None):
        if loading_path is None:
            loading_path = self.model_path
        self.model = SAC.load(loading_path)

    def predict(self, obs):
        action, _states = self.model.predict(obs)
        return action


if __name__ == "__main__":
    if args.alg == "ppo":
        savepath = "./" + "cnn_" + ENVS[args.env] + "_ppo.zip"
        train = PPOTrain(args.env, feature_extractor="cnn", model_path=savepath)
        train.train(int(10e7))
    elif args.alg == "sac":
        savepath = "./" + "cnn_" + ENVS[args.env] + "_sac.zip"
        train = SACTrain(args.env, feature_extractor="cnn", model_path=savepath)
        train.train(int(10e7), savepath)

    # path_3 = "./" + "cnn_" + ENVS[1] + "_her.zip"
    # train = HERTrain(1, feature_extractor="cnn", model_path=path_3)
    # train.train(10000, path_3)

    # path_0 = "./" + "cnn_" + ENVS[0] + "_sac.zip"
    # train = SACTrain(0, feature_extractor="cnn", model_path=path_0)
    # env = DoomEnv(display=True, feature='cnn', env_index=0, learning_type="sac")
    # obs = env.reset()
    # while True:
    #     sction = train.predict(obs=obs)
    #     obs, reward, done, info = env.step(sction, wait=True)
