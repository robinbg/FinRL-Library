# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
# from stable_baselines import SAC
# from stable_baselines import TD3
from stable_baselines3.td3.policies import TD3Policy
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback


Schedule = Callable[[float], float]

import gym
import torch as th
from torch import nn
from finrl.config import config


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    def forward(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

class CustomDDPG(DDPG):
    def __init__(
            self,
            policy: Union[str, Type[TD3Policy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = int(1e6),
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: int = -1,
            gradient_steps: int = -1,
            n_episodes_rollout: int = 1,
            action_noise: Optional[ActionNoise] = None,
            optimize_memory_usage: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Dict[str, Any] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super(DDPG, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            n_episodes_rollout=n_episodes_rollout,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=_init_setup_model,
        )

    def _sample_action(
            self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=True)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
    train_PPO()
        the implementation for PPO algorithm
    train_A2C()
        the implementation for A2C algorithm
    train_DDPG()
        the implementation for DDPG algorithm
    train_TD3()
        the implementation for TD3 algorithm      
    train_SAC()
        the implementation for SAC algorithm 
    DRL_prediction() 
        make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def train_A2C(self, model_name, model_params=config.A2C_PARAMS):
        """A2C model"""
        from stable_baselines3 import A2C
        env_train = self.env
        start = time.time()
        model = A2C('MlpPolicy', env_train,
                    n_steps=model_params['n_steps'],
                    ent_coef=model_params['ent_coef'],
                    learning_rate=model_params['learning_rate'],
                    verbose=model_params['verbose'],
                    tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name="A2C_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (A2C): ', (end - start) / 60, ' minutes')
        return model

    def train_DDPG(self, model_name, model_params=config.DDPG_PARAMS, using_default_policy=True):
        """DDPG model"""
        from stable_baselines3 import DDPG
        #  from stable_baselines.ddpg.policies import DDPGPolicy
        from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

        env_train = self.env

        n_actions = env_train.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        start = time.time()
        if using_default_policy:
             model = DDPG('MlpPolicy',
                     env_train,
                     batch_size=model_params['batch_size'],
                     buffer_size=model_params['buffer_size'],
                     param_noise=param_noise,
                     action_noise=action_noise,
                     verbose=model_params['verbose'],
                     tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                     )
        else:
             model = CustomDDPG('MlpPolicy',
                     env_train,
                     batch_size=model_params['batch_size'],
                     buffer_size=model_params['buffer_size'],
                     param_noise=param_noise,
                     action_noise=action_noise,
                     verbose=model_params['verbose'],
                     tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                     )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name="DDPG_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end - start) / 60, ' minutes')
        return model

    def train_TD3(self, model_name, model_params=config.TD3_PARAMS):
        """TD3 model"""
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise

        env_train = self.env

        n_actions = env_train.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        start = time.time()
        model = TD3('MlpPolicy', env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate=model_params['learning_rate'],
                    action_noise=action_noise,
                    verbose=model_params['verbose'],
                    tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name="TD3_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end - start) / 60, ' minutes')
        return model

    def train_SAC(self, model_name, model_params=config.SAC_PARAMS):
        """TD3 model"""
        from stable_baselines3 import SAC

        env_train = self.env

        start = time.time()
        model = SAC('MlpPolicy', env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate=model_params['learning_rate'],
                    learning_starts=model_params['learning_starts'],
                    ent_coef=model_params['ent_coef'],
                    verbose=model_params['verbose'],
                    tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name="SAC_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (SAC): ', (end - start) / 60, ' minutes')
        return model

    def train_PPO(self, model_name, model_params=config.PPO_PARAMS, using_default_policy=True):
        """PPO model"""
        from stable_baselines3 import PPO
        env_train = self.env

        start = time.time()
        if using_default_policy:
            model = PPO('MlpPolicy', env_train,
                        n_steps=model_params['n_steps'],
                        ent_coef=model_params['ent_coef'],
                        n_epochs=model_params['n_epochs'],
                        learning_rate=model_params['learning_rate'],
                        batch_size=model_params['batch_size'],
                        verbose=model_params['verbose'],
                        tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                        )
        else:
            model = PPO(CustomActorCriticPolicy, env_train,
                        n_steps=model_params['n_steps'],
                        ent_coef=model_params['ent_coef'],
                        n_epochs=model_params['n_epochs'],
                        learning_rate=model_params['learning_rate'],
                        batch_size=model_params['batch_size'],
                        verbose=model_params['verbose'],
                        tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                        )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name="PPO_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (PPO): ', (end - start) / 60, ' minutes')
        return model

    @staticmethod
    def DRL_prediction(model, test_data, test_env, test_obs):
        """make a prediction"""
        start = time.time()
        account_memory = []
        actions_memory = []
        for i in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs, deterministic=True)
            print(action)
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(test_data.index.unique()) - 2):
                account_memory = test_env.env_method(method_name='save_asset_memory')
                actions_memory = test_env.env_method(method_name='save_action_memory')
        end = time.time()
        return account_memory[0], actions_memory[0]
