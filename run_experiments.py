"""
Run multiple experiments on a single machine.
"""
import subprocess

import numpy as np

ALGOS = ["DDPG"]
TICKERS = 'PETR4.SA'

HMAX = [200, 500]
INITIAL_AMOUNT = 100000
N_TRIALS = 10

TUNING_TIMESTEPS = 30000
TRAIN_TIMESTEPS = 200000
N_REPEATS_BY_TRIAL = 3
METRIC_TO_OPTIMIZE = 'Sortino ratio'

TRAIN_PERIOD = ['2017-01-01', '2018-12-31']
EVAL_PERIOD = ['2019-01-01', '2019-12-31']
TRADE_PERIOD = ['2020-01-01', '2021-12-31']
USE_BEST_MODEL = True


USE_FUNDAMENTAL_INDICATORS = [False, True]
USE_TECH_INDICATORS = [True]

for alg in ALGOS:
    for hmax in HMAX:
        for use_technical in USE_TECH_INDICATORS:
            for use_fundamental in USE_FUNDAMENTAL_INDICATORS:
                OUTPUT_PATH = f"{alg}_{hmax}"
                OUTPUT_PATH += '_TECH' if use_technical else ''
                OUTPUT_PATH += '_FUND' if use_fundamental else ''
                args = [
                    "--alg",
                    alg,
                    "--tickers",
                    TICKERS,
                    "--hmax",
                    hmax,
                    "--train-period",
                    TRAIN_PERIOD[0], TRAIN_PERIOD[1],
                    "--eval-period",
                    EVAL_PERIOD[0], EVAL_PERIOD[1],
                    "--trade-period",
                    TRADE_PERIOD[0], TRADE_PERIOD[1],
                    "--initial-amount",
                    INITIAL_AMOUNT,
                    "--n-trials",
                    N_TRIALS,
                    "--metric-to-optimize",
                    METRIC_TO_OPTIMIZE,
                    "--train-timesteps",
                    TRAIN_TIMESTEPS,
                    "--tuning-timesteps",
                    TUNING_TIMESTEPS,
                    "--n-trials",
                    N_TRIALS,
                    "--n-repeats-by-trial",
                    N_REPEATS_BY_TRIAL,
                    "--use-best-model",
                    USE_BEST_MODEL,
                    "--output-path",
                    OUTPUT_PATH
                    
                ]
                if use_fundamental:
                    args += ["--use-fundamental-indicators", use_fundamental]
                if use_technical:
                    args += ["--use-tech-indicators", use_technical]
                args = list(map(str, args))
                print(["python", "tune_hyperparameters.py"] + args)
                ok = subprocess.call(["python", "tune_hyperparameters.py"] + args)
