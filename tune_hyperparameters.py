import argparse
from copy import deepcopy
import os
from pathlib import Path
import json
import itertools
import pandas as pd
import numpy as np
import optuna
from datetime import datetime

from sqlalchemy import column
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.plot import backtest_stats
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, sync_envs_normalization

from utils import model_utils, file_utils, data_processing_utils
from utils.model_utils import ALGORITHMS
from utils.callbacks import EvalCallback
import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--alg', help='RL Algorithm to train', default='PPO', 
        type=str, required=False, choices=list(ALGORITHMS.keys())
    )

    parser.add_argument(
        '--tickers', help='Tickers to download', default=['BOVA11.SA'], 
        required=False, nargs='+'
    )

    parser.add_argument(
        '--tuning-timesteps', help='Timesteps by trial', default=20000, 
        type=int, required=False
    )

    parser.add_argument(
        '--train-timesteps', help='Timesteps to training', default=200000, 
        type=int, required=False
    )

    parser.add_argument(
        '--train-period', nargs='+', required=False,
        default=[config.TRAIN_START_DATE, config.TRAIN_END_DATE]
        
    )
    parser.add_argument(
        '--eval-period', nargs='+', required=False,
        default=[config.EVAL_START_DATE, config.EVAL_END_DATE]
    )
    parser.add_argument(
        '--trade-period', nargs='+', required=False,
        default=[config.TRADE_START_DATE, config.TRADE_END_DATE]
    )

    parser.add_argument("--hmax", default=200, type=int)

    parser.add_argument("--use-ohlcv", default=False, type=bool)
    
    # parser.add_argument("--random-initial-day", default=False, type=bool)

    # parser.add_argument("--use-turbulence-as-feature", default=False, type=bool)

    parser.add_argument("--initial-amount", default=100000, type=int)

    parser.add_argument("--n-trials", default=30, type=int)
    
    parser.add_argument("--n-repeats-by-trial", default=1, type=int)

    parser.add_argument('--output-path', default='', type=str)

    parser.add_argument('--metric-to-optimize', default='Cumulative returns', type=str)

    parser.add_argument('--n-eval-episodes', default=1, type=int)
    
    parser.add_argument('--use-best-model', default=False, type=bool)

    parser.add_argument('--use-fundamental-indicators', default=False, type=bool)

    parser.add_argument('--use-tech-indicators', default=False, type=bool)

    parser.add_argument(
        '--indicators', nargs='+', default=config.INDICATORS,
        required=False
    )


    args = parser.parse_args()

    print('-'*100)
    print('Starting tuning at {}'.format(datetime.now()))
    print('-'*100, '\n')

    # MAKING FOLDERS
    if len(args.tickers) > 4:
        TICKER_NAME = 'MULTIPLE_TICKERS'
    else:
        TICKER_NAME = '_'.join(args.tickers).replace('.SA', '')

    if args.output_path == '':
        OUTPUT_PATH = './TUNING/{}/{}_HMAX_{}'.format(
            TICKER_NAME, args.alg, args.hmax
        )

        OUTPUT_PATH = file_utils.uniquify(OUTPUT_PATH)
    else:
        OUTPUT_PATH = os.path.join(
            './TUNING', TICKER_NAME, args.output_path
        )

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'monitor_logs'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'actions_logs'), exist_ok=True)

    with open(os.path.join(OUTPUT_PATH, 'command_line_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # DATA PROCESSING
    tech_indicators = []
    
    if args.use_fundamental_indicators:
        df = data_processing_utils.get_data_with_fundamentals(
            start_date=args.train_period[0], 
            end_date=args.trade_period[1], 
            tickers=args.tickers,
            tech_indicators=args.indicators
        )
        tech_indicators += [
            'LPA', 'VPA', 'P/L', 'P/EBITDA', 'P/VPA', 'DL/PL', 'DL/EBITDA',
            'ROE', 'MARGEM_EBITDA', 'DL/EBIT', 'MARGEM_EBIT', 'MARGEM_LIQUIDA'
        ]
    else:
        df = data_processing_utils.get_data(
            start_date=args.train_period[0], 
            end_date=args.trade_period[1], 
            tickers=args.tickers,
            tech_indicators=args.indicators
        )

    if args.use_tech_indicators:
        tech_indicators += args.indicators


    # STORING DATASET
    df.to_pickle(
        os.path.join(
            OUTPUT_PATH,
            'dataset.pkl'
        )
    )

    print(df.tail())

    # ENVIRONMENT CONFIGS
    
    if args.use_ohlcv: 
        tech_indicators += ['open', 'high', 'low', 'volume']
    
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(tech_indicators)*stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension

    num_stock_shares = [0] * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    print(args)
    print(args.hmax)

    env_train_kwargs = {
        'stock_dim': stock_dimension,
        'hmax': args.hmax,
        'initial_amount': args.initial_amount,
        'num_stock_shares': num_stock_shares,
        'buy_cost_pct': buy_cost_list,
        'sell_cost_pct': sell_cost_list,
        'reward_scaling': 1,
        'state_space': state_space,
        'action_space': stock_dimension,
        'tech_indicator_list': tech_indicators,
        'turbulence_threshold': None,
        'print_verbosity': 1000
    }


    with open(os.path.join(OUTPUT_PATH, 'env_train_kwargs.json'), 'w') as f:
        json.dump(env_train_kwargs, f, indent=4)

    
    # HIPERPARAMETER TUNING
    train_set = data_split(df, args.train_period[0], args.train_period[1])
    eval_set = data_split(df, args.eval_period[0], args.eval_period[1])

    print('-'*100)
    print('Buy and Hold Cummulative Returns:')
    print('Train Set:', train_set.iloc[-1].close / train_set.iloc[0].close - 1)
    print('Eval Set:', eval_set.iloc[-1].close / eval_set.iloc[0].close - 1)
    print('-'*100)



    def objective(trial: optuna.Trial):        
       
        trial.n_actions = len(args.tickers)

        # Optimize buffer size, batch size, learning rate
        hyperparameters = model_utils.get_params_grid(
            args.alg, trial
        )
        initial_hyperparameters = deepcopy(hyperparameters)
        print(initial_hyperparameters)
        
        metric = 0
        count_zero_variance = 0
        for n_repeat in range(args.n_repeats_by_trial):
            
            # env_train_kwargs['num_stock_shares'] = np.random.randint(0, args.hmax, stock_dimension).tolist()

            e_train_gym = StockTradingEnv(
                df=train_set, 
                **env_train_kwargs
            )

            env_train = DummyVecEnv([lambda: e_train_gym])
            env_train = VecMonitor(
                env_train, 
                os.path.join(
                    OUTPUT_PATH,
                    'monitor_logs',
                    'log_train_{}_{}'.format(str(trial.number), str(n_repeat))
                )
            )

            env_train = VecNormalize(
                env_train, 
                training=True,
                norm_obs=True, 
                norm_reward=True,
                gamma=hyperparameters['gamma']
            )

            # ENV EVAL
            env_eval_kwargs = env_train_kwargs.copy()
            # if 'random_initial_day' in env_eval_kwargs.keys():
            #     del env_eval_kwargs['random_initial_day']

            # scaled_eval_set = pd.concat([
            #     eval_set[not_num_columns], 
            #     pd.DataFrame(
            #         scaler.transform(eval_set.drop(not_num_columns, axis=1)),
            #         index=eval_set.index,
            #         columns=eval_set.drop(not_num_columns, axis=1).columns
            #     )], axis=1
            # )

            e_eval_gym = StockTradingEnv(df = eval_set, **env_eval_kwargs)
            env_eval = DummyVecEnv([lambda: e_eval_gym])
            
            PATH_LOG_EVAL = os.path.join(
                OUTPUT_PATH, 
                'monitor_logs',
                'log_eval_{}_{}'.format(str(trial.number), str(n_repeat))
            )

            env_eval = VecMonitor(
                env_eval, 
                PATH_LOG_EVAL
            )

            env_eval = VecNormalize(
                env_eval, 
                training=False,
                norm_obs=True, 
                norm_reward=False,
                gamma=hyperparameters['gamma']
            )

            eval_callback = EvalCallback(
                env_eval, 
                n_eval_episodes=args.n_eval_episodes,
                eval_freq=int(train_set.shape[0]/len(args.tickers)),
                # log_path=os.path.join(
                #     OUTPUT_PATH,
                #     f'log_callback_{trial.number}'
                # ),
                best_model_save_path=os.path.join(
                    OUTPUT_PATH,
                    f'models',
                    'TRIAL_{}_REPEAT_{}'.format(trial.number, n_repeat)
                ),
                verbose=False
            )
            
            MODEL = model_utils.get_model(args.alg)
            model = MODEL(
                policy='MlpPolicy', 
                env=env_train,
                **hyperparameters,
                verbose=False,
                seed=None,
                tensorboard_log=os.path.join(
                    OUTPUT_PATH,
                    './tb_log'
                )
            )
            
            #You can increase it for better comparison
            try:
                trained_model = model.learn(
                    tb_log_name='{}_{}'.format(
                        args.alg, str(trial.number) + '_' + str(n_repeat)
                    ),
                    log_interval=1,
                    total_timesteps=args.tuning_timesteps,
                    callback=[eval_callback],
                    reset_num_timesteps=True
                )
            except ValueError as e:
                print(e)
                print('Interrupting execution of trial {}'.format(trial.number))
                return -np.inf



            model_path = os.path.join(
                OUTPUT_PATH, 
                'models', 
                'TRIAL_{}_REPEAT_{}'.format(trial.number, n_repeat)
            )
            trained_model.save(os.path.join(model_path, 'model_final.zip'))
            if trained_model.get_vec_normalize_env() is not None:
                trained_model.env.save(os.path.join(model_path, 'env_statistics_final'))

            # EVALUATE MODEL

            e_eval_gym = StockTradingEnv(df=eval_set, **env_eval_kwargs)
            env_eval = DummyVecEnv([lambda: e_eval_gym])
            if args.use_best_model:
                env_statistics_path = os.path.join(
                    OUTPUT_PATH, 
                    'models',
                    'TRIAL_{}_REPEAT_{}'.format(trial.number, n_repeat),
                    'env_statistics_best'
                )

                if Path(env_statistics_path).exists():
                    env_eval = VecNormalize.load(
                        env_statistics_path,
                        env_eval
                    )
                    env_eval.training = False
                    env_eval.norm_reward = False

                best_model_path = os.path.join(
                    OUTPUT_PATH, 
                    f'models',
                    'TRIAL_{}_REPEAT_{}'.format(trial.number, n_repeat),
                    'model_best.zip'
                )

                MODEL = model_utils.get_model(args.alg)
                trained_model = MODEL.load(best_model_path, env_eval)

                print('USANDO BEST MODEL')
                print('TIMESTEPS:', trained_model.num_timesteps)
            else:
                trained_model = MODEL(
                    policy='MlpPolicy', 
                    env=env_train,
                    **hyperparameters,
                    verbose=True,
                    seed=None,
                    tensorboard_log=os.path.join(
                        OUTPUT_PATH,
                        './tb_log'
                    )
                )

            account_memory, actions_memory, state_memory = model_utils.predict(trained_model, env_eval)

            if account_memory['account_value'].std() == 0:
                count_zero_variance += 1

            if count_zero_variance >= 2:
                return -np.inf

            temp_metric = backtest_stats(account_value=account_memory)[
                args.metric_to_optimize
            ]

            df_history = account_memory.merge(actions_memory, how='left', on='date')
            df_history = df_history.merge(eval_set[['date', 'close']], how='left', on='date')
            df_history['close_return'] = df_history['close'].pct_change(1)
            df_history['cumulated_close'] = (df_history['close_return'] + 1).cumprod()
            df_history['daily_return'] = df_history['account_value'].pct_change(1)
            df_history['cumulated_return'] = (df_history['daily_return'] + 1).cumprod()
            df_history = df_history.merge(state_memory, how='left', on='date')

            # df_log_eval = pd.read_csv(
            #     PATH_LOG_EVAL + '.monitor.csv',
            #     skiprows=[0]
            # )
            # print(df_log_eval)

            # temp_metric = df_log_eval.tail(args.n_eval_episodes * 5)['r'].mean()
            # if temp_metric == 0:
            #     return -np.inf

            # print(account_memory)
            # print('Last Value:', account_memory.iloc[-1])
            # temp_metric = account_memory['account_value'].iloc[-1] / account_memory['account_value'].iloc[0] - 1


            register = dict()
            
            register['hyperparameters'] = model_utils.convert_params_to_store_format(
                deepcopy(initial_hyperparameters)
            )
            
            register['trial'] = trial.number
            register['metric'] = temp_metric
            

            hp_hist_path = os.path.join(OUTPUT_PATH, f'./hp_tuning_hist.csv')
            if Path(hp_hist_path).exists():
                header=False
            else:
                header=True

            df_register = pd.DataFrame([register])
            df_register = df_register[['trial', 'metric', 'hyperparameters']]
            df_register.to_csv(
                hp_hist_path,
                mode='a',
                index=False,
                header=header
            )

            df_history.to_csv(
                os.path.join(
                    OUTPUT_PATH, 
                    'actions_logs', 
                    'eval_history_{}_{}.csv'.format(
                        trial.number, n_repeat
                    )
                )
            )

            metric += temp_metric

        metric_mean = metric/args.n_repeats_by_trial

        # print('\nTrial {} - Cumulative Return: {:.4f}'.format(
        #     trial.number, metric_mean
        # ))
        # # Calculate trade performance metric
        # # Currently ratio of average win and loss market values
        # tpm = calc_trade_perf_metric(df_actions,trade,tp_metric)
        return metric_mean

    #Create a study object and specify the direction as 'maximize'
    #As you want to maximize sharpe
    #Pruner stops not promising iterations
    #Use a pruner, else you will get error related to divergence of model
    #You can also use Multivariate samplere
    #sampler = optuna.samplers.TPESampler(multivarite=True,seed=42)

    sampler = optuna.samplers.TPESampler(
        seed=225214
    )

    study = optuna.create_study(
        storage='sqlite:///{}/{}'.format(OUTPUT_PATH, 'optuna.db'),
        study_name="{}_study".format(args.alg),
        direction='maximize',
        sampler=sampler, 
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=True
    )

    # logging_callback = optuna_callbacks.LoggingCallback(threshold=lc_threshold,
    #                                    patience=lc_patience,
    #                                    trial_number=lc_trial_number)

    #You can increase the n_trials for a better search space scanning
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        catch=(ValueError,)#,
        # callbacks=[logging_callback]
    )