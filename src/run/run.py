import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
import copy

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
    map_list = copy.deepcopy(args.map_list)
    # for grf tasks
    # numagents_list = copy.deepcopy(args.numagents_list)
    total_map_num = len(map_list)
    
    logger_dict = {}
    args_dict = {}
    runner_dict = {}
    env_info_dict = {}
    scheme_dict = {}
    groups_dict = {}
    preprocess_dict = {}
    buffer_dict = {}
    buffer_scheme_dict = {}
    
    for id, map_name in enumerate(map_list):
        logger_dict[id] = copy.deepcopy(logger)
        args_dict[id] = copy.deepcopy(args)
        args_dict[id].env_args['map_id'] = id
        args_dict[id].env_args['map_name'] = map_name

        # for grf tasks
        # args_dict[id].env_args['num_agents'] = int(numagents_list[id])

        runner_dict[id] = r_REGISTRY[args.runner](args=args_dict[id], logger=logger_dict[id])
        
        env_info_dict[id] = runner_dict[id].get_env_info()
        args_dict[id].n_agents = env_info_dict[id]["n_agents"]
        args_dict[id].n_actions = env_info_dict[id]["n_actions"]
        args_dict[id].state_shape = env_info_dict[id]["state_shape"]
        args_dict[id].accumulated_episodes = getattr(args, "accumulated_episodes", None)

        if getattr(args_dict[id], 'agent_own_state_size', False):
            args_dict[id].agent_own_state_size = get_agent_own_state_size(args_dict[id].env_args)

        scheme_dict[id] = {
            'map_id': {"vshape": (1,)}, # 地图ID
            "state": {"vshape": env_info_dict[id]["state_shape"]},
            "obs": {"vshape": env_info_dict[id]["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info_dict[id]["n_actions"],), "group": "agents", "dtype": th.int},
            "probs": {"vshape": (env_info_dict[id]["n_actions"],), "group": "agents", "dtype": th.float},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups_dict[id] = {
                "agents": args_dict[id].n_agents
        }
        preprocess_dict[id] = {
                "actions": ("actions_onehot", [OneHot(out_dim=args_dict[id].n_actions)])
        }
        buffer_dict[id] = ReplayBuffer(scheme_dict[id], groups_dict[id], args_dict[id].buffer_size, 
                          env_info_dict[id]["episode_limit"] + 1, preprocess=preprocess_dict[id],
                          device="cpu" if args_dict[id].buffer_cpu_only else args_dict[id].device)   
        buffer_scheme_dict[id] = buffer_dict[id].scheme

    mac = mac_REGISTRY[args_dict[0].mac](buffer_scheme_dict, groups_dict, args_dict, total_map_num)

    for id, map_name in enumerate(map_list):
        runner_dict[id].setup(scheme=scheme_dict[id], groups=groups_dict[id], preprocess=preprocess_dict[id], mac=mac)
    
    learner = le_REGISTRY[args.learner](mac, logger_dict, args_dict, total_map_num)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    episode_list = [0 for _ in range(total_map_num)]
    last_test_T_list = [(-args.test_interval - 1) for _ in range(total_map_num)]
    last_log_T_list = [(-args.test_interval - 1) for _ in range(total_map_num)]
    model_save_time_list = [0 for _ in range(total_map_num)]

    start_time = time.time()
    last_time_list = [start_time for _ in range(total_map_num)]

    logger_dict[0].console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    loss_backward_flag = False
    runner_dict_t_env = {}
    runner_dict_battle_won = [0]*total_map_num
    the_slowest_runner_step = 0
    while the_slowest_runner_step <= args.t_max:

        # Run for a whole episode at a time
        with th.no_grad():
            for i in range(total_map_num):
                episode_batch = runner_dict[i].run(test_mode=False)
                buffer_dict[i].insert_episode_batch(episode_batch)

        if not loss_backward_flag:
            for i in range(total_map_num):
                if not buffer_dict[i].can_sample(args.batch_size):
                    break
                loss_backward_flag = True
        
        if loss_backward_flag:
            episode_sample_dict = {}
            for i in range(total_map_num):
                next_episode = episode + args.batch_size_run
                if args_dict[i].accumulated_episodes and next_episode % args_dict[i].accumulated_episodes != 0:
                    continue
                episode_sample = buffer_dict[i].sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                episode_sample_dict[i] = episode_sample
                runner_dict_t_env[i] = runner_dict[i].t_env
                runner_dict_battle_won[i] = runner_dict[i].battle_won
            learner.train(episode_sample_dict, runner_dict_t_env, runner_dict_battle_won, episode)
            del episode_sample_dict
            # th.cuda.empty_cache()
            
        for i in range(total_map_num):
            n_test_runs = max(1, args.test_nepisode // runner_dict[i].batch_size)
            if (runner_dict[i].t_env - last_test_T_list[i]) / args.test_interval >= 1.0:

                logger_dict[i].console_logger.info("env_name: {} t_env: {} / {}".format(args_dict[i].env_args['map_name'], runner_dict[i].t_env, args.t_max))
                logger_dict[i].console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time_list[i], last_test_T_list[i], runner_dict[i].t_env, args.t_max), time_str(time.time() - start_time)))
                last_time_list[i] = time.time()

                last_test_T_list[i] = runner_dict[i].t_env
                for _ in range(n_test_runs):
                    runner_dict[i].run(test_mode=True)
        
                if args.save_model and (runner_dict[i].t_env - model_save_time_list[i] >= args.save_model_interval):
                # if args.save_model and (runner_dict[i].t_env - model_save_time_list[i] >= args.save_model_interval or model_save_time_list[i] == 0):
                    model_save_time_list[i] = runner_dict[i].t_env
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, args_dict[i].env_args['map_name'], str(runner_dict[i].t_env))
                    #"results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    logger_dict[i].console_logger.info("Saving models to {}".format(save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    learner.save_models(save_path, i)

                episode_list[i] += args.batch_size_run
                if (runner_dict[i].t_env - last_log_T_list[i]) >= args.log_interval:
                    logger_dict[i].log_stat("episode", episode_list[i], runner_dict[i].t_env)
                    logger_dict[i].print_recent_stats()
                    last_log_T_list[i] = runner_dict[i].t_env
        
        episode += args.batch_size_run

        the_slowest_runner_step = runner_dict[0].t_env
        for i in range(1, total_map_num):
            if runner_dict[i].t_env < the_slowest_runner_step:
                the_slowest_runner_step = runner_dict[i].t_env

    for i in range(total_map_num):
        runner_dict[i].close_env()
    logger_dict[0].console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
