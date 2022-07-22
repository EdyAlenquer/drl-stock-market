import stable_baselines3 as sb3
from stable_baselines3 import A2C, SAC, DDPG, PPO, TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


import optuna
from typing import Dict, Any, Union, Callable
import numpy as np
import torch
from torch import nn


ALGORITHMS = {
    'DDPG': DDPG,
    'SAC': SAC,
    'PPO': PPO,
    'TD3': TD3,
}


def get_model(alg):
    return ALGORITHMS[alg]


def convert_params_to_store_format(dict):
    dict = dict.copy()
    
    for key in dict.keys():
        if isinstance(dict, type(dict[key])):
            for key_2 in dict[key].keys():
                if key_2 == 'activation_fn':
                    dict[key][key_2] = dict[key][key_2].__name__
        else:
            if key == 'action_noise':
                action_noise_mean = dict[key]._mu
                action_noise_sigma = dict[key]._sigma
                dict[key] = dict[key].__class__.__name__

    if 'action_noise' in dict:
        dict['action_noise_mean'] = list(action_noise_mean)
        dict['action_noise_sigma'] = list(action_noise_sigma)

    return dict


def load_params_from_store_format(dict):
    dict = dict.copy()
    
    for key in dict.keys():
        if isinstance(dict, type(dict[key])):
            for key_2 in dict[key].keys():
                if key_2 == 'activation_fn':
                    dict[key][key_2] = getattr(torch.nn, dict[key][key_2])
        else:
            if key == 'action_noise':
                dict[key] = getattr(sb3.common.noise, dict[key])(
                    mean=dict['action_noise_mean'],
                    sigma=dict['action_noise_sigma']
                )
    
    if 'action_noise' in dict:
        del dict['action_noise_mean']
        del dict['action_noise_sigma']
    
    return dict


def get_params_grid(alg, trial):
    if alg == 'DDPG':
        return sample_ddpg_params(trial)
    elif alg == 'PPO':
        return sample_ppo_params(trial)
    elif alg == 'SAC':
        return sample_sac_params(trial)
    elif alg == 'TD3':
        return sample_td3_params(trial)


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 10000, 20000])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

    return hyperparams


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    # clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    # n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    # gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    # max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        # "ent_coef": ent_coef,
        # "clip_range": clip_range,
        # "n_epochs": n_epochs,
        # "gae_lambda": gae_lambda,
        # "max_grad_norm": max_grad_norm,
        # "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999, ])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    # tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    # train_freq = trial.suggest_categorical("train_freq", [(8, 64, 128, 256, 512])
    # train_freq_type = trial.suggest_categorical('train_freq_type', ['episode', 'step'])
    
    # if train_freq_type == 'episode':
    # train_freq = trial.suggest_categorical("train_freq", [1, 2, 3])
    # elif train_freq_type == 'step':
    train_freq = trial.suggest_categorical("train_freq", [8, 256, 1024])

    # train_freq = trial.suggest_categorical("train_freq", [1])
    gradient_steps = trial.suggest_categorical("gradient_steps", [8, 64, 256])

    # noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    # noise_std = trial.suggest_uniform("noise_std", 0, 0.2)

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]


    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        # "tau": tau,
        "learning_starts": 20000, 
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": (train_freq, 'step'),
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(
            net_arch=net_arch,
            # activation_fn=activation_fn
        ),
    }

    # if noise_type == "normal":
    #     hyperparams["action_noise"] = NormalActionNoise(
    #         mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
    #     )
    # elif noise_type == "ornstein-uhlenbeck":
    #     hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
    #         mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
    #     )

    # if trial.using_her_replay_buffer:
    #     hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 0.3)

    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "verybig": [256, 256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
        "tau": tau,
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    # if trial.using_her_replay_buffer:
    #     hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def predict(model, env, deterministic=True):
    state = env.reset()
    n_tickers = len(env.envs[0].df['tic'].unique())
    N = int(env.envs[0].df.shape[0] / n_tickers)
    for j in range(N):
        action, _states = model.predict(state, deterministic)
        state, rewards, dones, info = env.step(action)
        if j == (N - 2):
            account_memory = env.env_method(method_name="save_asset_memory")[0]
            actions_memory = env.env_method(method_name="save_action_memory")[0]
            state_memory = env.env_method(method_name="save_state_memory")[0]
        if dones:
            break
    return account_memory, actions_memory, state_memory

