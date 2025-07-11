import argparse
import datetime
import importlib
import os
import sys
import time

import gym as openai_gym
import gymnasium
import numpy as np
import tensorboardX
import torch
import torch_ac
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from syllabus.core import GymnasiumSyncWrapper, make_multiprocessing_curriculum, Evaluator, GymnasiumEvaluationWrapper
from syllabus.curricula import LearningProgress, OnlineLearningProgress, OMNI, interestingness_from_json, StratifiedLearningProgress, StratifiedOnlineLearningProgress, Learnability, OMNILearnability, CentralPrioritizedLevelReplay, StratifiedDomainRandomization, StratifiedLearnability, DomainRandomization, SequentialCurriculum
from syllabus.task_space import DiscreteTaskSpace
from torch_ac.utils import ParallelEnv

import utils
import wandb
from envs import env_tr_uni, env_tr_syllabus, env_tr_syllabus_seed
from envs.env_utils import ach_to_string
from envs.syllabus_wrapper import CrafterTaskWrapper, CrafterSeedWrapper
from model import ACModel
from utils import device, preprocess_images, preprocess_totensor
from utils.prep_train import get_rdn_tsr


def preprocessor(obss, device=device):
    prep_obss = dict()
    for key in obss.keys():
        if key == 'image':
            prep_obss[key] = preprocess_images([obs for obs in obss['image']], device=device)
        else:
            prep_obss[key] = preprocess_totensor([obs for obs in obss[key]], device=device)
    return torch_ac.DictList(prep_obss)


class ACEvaluator(Evaluator):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)

    def _get_action(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        if self.agent.recurrent:
            dist, _, lstm_state = self.agent(state, lstm_state * (1 - done).unsqueeze(1))
        else:
            dist, _ = self.agent(state)
        action = dist.sample()
        return action, lstm_state, {}

    def _get_value(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        if self.agent.recurrent:
            _, value, lstm_state = self.agent(state, lstm_state * (1 - done).unsqueeze(1))
        else:
            _, value = self.agent(state)
        return value, lstm_state, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        if self.agent.recurrent:
            dist, value, lstm_state = self.agent(state, lstm_state * (1 - done).unsqueeze(1))
        else:
            dist, value = self.agent(state)
        action = dist.sample()
        return action, value, lstm_state, {}

    def _check_inputs(self, lstm_state, done):
        assert (
            lstm_state is not None
        ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        assert (
            done is not None
        ), "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        return True

    def _prepare_state(self, state):

        if self.preprocess_obs is not None:
            state = self.preprocess_obs(state)
        return state

    def _prepare_lstm(self, lstm_state, done):
        lstm_state = torch.Tensor(lstm_state).to(self.device)
        done = torch.Tensor(done).to(self.device)
        return lstm_state, done

    def _set_eval_mode(self):
        self.agent.eval()

    def _set_train_mode(self):
        self.agent.train()


def eval_all_tasks(acmodel, penv, num_eps=1, wrap=False):
    def thunk(eval_episodes=1):
        given_counts = np.zeros(len(penv.envs[0].given_achievements))
        follow_counts = np.zeros(len(penv.envs[0].follow_achievements))
        agent = utils.Agent.model_init(penv.observation_space, acmodel, num_envs=len(penv.envs))
        acmodel.eval()
        with torch.no_grad():
            obss = penv.reset()
            ep_counter = 0
            while ep_counter < eval_episodes:
                actions = agent.get_actions(obss)
                obss, rewards, terminateds, truncateds, infos = penv.step(actions)
                dones = tuple(a | b for a, b in zip(terminateds, truncateds))

                for i, done in enumerate(dones):
                    if done:
                        given_counts += list(infos[i]['given_achs'].values())
                        follow_counts += list(infos[i]['follow_achs'].values())
                        ep_counter += 1
                        if ep_counter % 100 == 0:
                            print([f"{f}/{g}" for f, g in zip(follow_counts, given_counts)])

        acmodel.train()
        follow_counts = np.concatenate((follow_counts, np.zeros(len(given_counts) - len(follow_counts))))
        task_success_rates = np.divide(follow_counts, given_counts,
                                       out=np.zeros_like(follow_counts), where=given_counts != 0)
        print(np.min(task_success_rates),
              np.mean(task_success_rates), np.max(task_success_rates))
        return task_success_rates
    if wrap:
        return thunk
    return thunk(eval_episodes=num_eps)


def make_task_env(curriculum=None, is_eval=False):
    def thunk():
        env = env_tr_syllabus.Env(eval_mode=is_eval, dummy_bits=args.dummy_bits)
        env = CrafterTaskWrapper(env, simple_task_space=args.simple_task_space)

        if is_eval:
            env = GymnasiumEvaluationWrapper(env, change_task_on_completion=True, start_index_spacing=1,
                                             eval_only_n_tasks=env.task_space.num_tasks)
        else:
            env = GymnasiumSyncWrapper(env, env.task_space, curriculum.components,
                                       buffer_size=1, change_task_on_completion=True)
        return env
    return thunk


def make_seed_env(curriculum=None, is_eval=False):
    def thunk():
        env = env_tr_syllabus_seed.Env(static_task=args.static_task)
        env = CrafterSeedWrapper(env)

        if is_eval:
            env = GymnasiumEvaluationWrapper(env, ignore_seed=True)
        else:
            env = GymnasiumSyncWrapper(env, env.task_space, curriculum.components, buffer_size=0)
        return env
    return thunk


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use: a2c | ppo")
    parser.add_argument("--env", default='custom',
                        help="name of the environment to train on")
    parser.add_argument("--exp-name", default='testing',
                        help="name of the experiment in wandb")
    parser.add_argument("--logging-dir", default='.',
                        help="directory to log wandb results")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="number of updates between two logs (default: 50)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="number of updates between two saves (default: 50, 0 means no saving)")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="number of updates between two evals (default: 25, 0 means no evaluating)")
    parser.add_argument("--procs", type=int, default=32,
                        help="number of processes (default: 32)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")
    # Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="batch size for PPO (default: 2048)")
    parser.add_argument("--frames-per-proc", type=int, default=1024,
                        help="number of frames per process before update (default: 5 for A2C and 1024 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.0001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ac-size", type=int, default=128,
                        help="actor-critic layer size (default: 128)")
    parser.add_argument("--activation", default='tanh',
                        help="activation to use: tanh | relu")

    # Syllabus arguments
    parser.add_argument("--syllabus", type=bool, default=False, help="use curriculum learning")
    parser.add_argument("--curriculum-method", type=str, default="learning_progress",
                        help="method to use for curriculum learning")
    parser.add_argument("--seed-curriculum", type=bool, default=False, help="use seed curriculum")
    parser.add_argument("--static-task", type=str, default="make_wood_pickaxe", help="static task for seed curriculum")
    parser.add_argument("--eval-num", type=int, default=20,
                        help="number of times to evaluate each task for learning progress (default: 20)")
    parser.add_argument("--eval-override", type=int, default=0,
                        help="number of episodes to evaluate each task (default: 0)")
    parser.add_argument("--full-eval", type=bool, default=False,
                        help="override and force full evaluation")
    parser.add_argument("--dummy-bits", type=int, default=10,
                        help="number of dummy bits for impossible tasks")
    parser.add_argument("--no-encode-task", action="store_true", default=False,
                        help="encode task in observation")
    parser.add_argument("--task-dropout", type=float, default=0.0,
                        help="probability of dropping task encodings during training (default: 0.0)")
    parser.add_argument("--simple-task-space", type=bool, default=False,
                        help="use a simple task space with only the base achievements (default: False)")

    # Parameters for learning progress
    parser.add_argument("--eval-procs", type=int, default=20,
                        help="number of processes (default: 20)")
    parser.add_argument("--ema-alpha", type=float, default=0.1,
                        help="smoothing value for ema in claculating learning progress (default: 0.1)")
    parser.add_argument("--p-theta", type=float, default=0.1,
                        help="parameter for reweighing learning progress (default: 0.1)")
    parser.add_argument("--online-uniform-prob", type=float, default=0.0,
                        help="probability of sampling uniformly from the task space in online learning progress (default: 0.0)")
    parser.add_argument("--online-save-last", type=bool, default=False,
                        help="whether to save the last success rates in online learning progress (default: False)")
    parser.add_argument("--normalize-success", type=bool, default=True,
                        help="whether to normalize success rates in learning progress (default: True)")
    parser.add_argument("--selection_metric", type=str, default="success",
                        help="metric to use for selection in learning progress (default: success, options: success, score, learnability)")

    # Learnability arguments
    parser.add_argument("--learnability-prob", type=float, default=1.0,
                        help="probability of sampling from the learnability curriculum (default: 1.0)")
    parser.add_argument("--learnability-topk", type=int, default=10,
                        help="number of top k tasks to select for sampling")

    # PLR arguments
    parser.add_argument("--plr-temperature", type=float, default=0.1,
                        help="temperature for PLR sampling (default: 0.1)")
    parser.add_argument("--plr-staleness-coef", type=float, default=0.1,
                        help="staleness coefficient for PLR sampling (default: 0.1)")
    args = parser.parse_args()
    args.recurrence = 2
    args.encode_task = not args.no_encode_task

    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project="syllabus-testing",
        entity="ryansullivan",
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        dir=args.logging_dir
    )

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_{args.exp_name}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = os.path.join(args.logging_dir, utils.get_model_dir(model_name))

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    # csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {
            "num_frames": 0, "update": 0,
            "p_fast": None, "p_slow": None, "raw_tsr": None, "ema_tsr": None,
        }
    txt_logger.info("Training status loaded\n")

    sample_env = env_tr_uni.Env(dummy_bits=args.dummy_bits)
    sample_env.reset()
    task_env = CrafterSeedWrapper(sample_env) if args.seed_curriculum else CrafterTaskWrapper(sample_env, simple_task_space=args.simple_task_space)

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(sample_env.observation_space)
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, sample_env.action_space,
                      acsize=args.ac_size, activation=args.activation, encode_task=args.encode_task, task_dropout=args.task_dropout)
    print(sum(p.numel() for p in acmodel.parameters() if p.requires_grad))

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    eval_eps = args.eval_num * \
        (task_env.task_space.num_tasks if args.seed_curriculum else len(
            sample_env.target_achievements) * 300 / sample_env._length)
    eval_eps = args.eval_override if args.eval_override != 0 else eval_eps
    print("Eval eps:", eval_eps)

    # Eval envs
    env_module = importlib.import_module(f'envs.env_{args.env}')
    eval_envs = [env_tr_uni.Env(dummy_bits=args.dummy_bits) for _ in range(args.eval_procs)]
    eval_envs = ParallelEnv(eval_envs)
    eval_envs.reset()
    eval_envs.set_curriculum(train=False)
    make_env = make_seed_env if args.seed_curriculum else make_task_env

    # Create curriculum
    if args.syllabus:
        sample_env = env_tr_syllabus_seed.Env(
            dummy_bits=args.dummy_bits) if args.seed_curriculum else env_tr_syllabus.Env(dummy_bits=args.dummy_bits)
        sample_env = CrafterSeedWrapper(sample_env) if args.seed_curriculum else CrafterTaskWrapper(sample_env, simple_task_space=args.simple_task_space)
        sample_env.reset()
        evaluator = ACEvaluator(acmodel, preprocess_obs=preprocessor, device=device)
        syllabus_eval_envs = AsyncVectorEnv([make_env(is_eval=True) for _ in range(args.eval_procs)])
        names = [f'{name}' for name in sample_env.task_space.tasks]
        print(sample_env.task_space.num_tasks)
        # print(sample_env.task_space.decode(105))

        def task_names(task, idx):
            return names[idx]

        # curriculum = LearningProgress(
        interestingness = interestingness_from_json('./moi_saved/preds_r.json')
        if args.curriculum_method == "dr":
            curriculum = DomainRandomization(
                sample_env.task_space,
                task_names=task_names,
            )
        elif args.curriculum_method == "learning_progress":
            curriculum = LearningProgress(
                sample_env.task_space,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                ema_alpha=args.ema_alpha,
                p_theta=args.p_theta,
                normalize_success=args.normalize_success,
            )
        elif args.curriculum_method == "online_learning_progress":
            curriculum = OnlineLearningProgress(
                sample_env.task_space,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                task_names=task_names,
                ema_alpha=args.ema_alpha,
                p_theta=args.p_theta,
                uniform_prob=args.online_uniform_prob,
                normalize_success=args.normalize_success,
                random_start_tasks=500,
                use_live_dist=True,
            )
        elif args.curriculum_method == "stratified_learning_progress":
            curriculum = StratifiedLearningProgress(
                sample_env.task_space,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                normalize_success=args.normalize_success,
                selection_metric=args.selection_metric,
            )
        elif args.curriculum_method == "stratified_online_learning_progress":
            curriculum = StratifiedOnlineLearningProgress(
                sample_env.task_space,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                task_names=task_names,
                ema_alpha=args.ema_alpha,
                p_theta=args.p_theta,
                uniform_prob=args.online_uniform_prob,
                normalize_success=args.normalize_success,
                random_start_tasks=500,
                use_live_dist=True,
                selection_metric=args.selection_metric,
            )
        elif args.curriculum_method == "omni":
            curriculum = OMNI(
                sample_env.task_space,
                interestingness=interestingness,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                normalize_success=args.normalize_success,
            )
        elif args.curriculum_method == "learnability":
            curriculum = Learnability(
                sample_env.task_space,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                sampling="dist",
                normalize_success=args.normalize_success)
        elif args.curriculum_method == "learnability_topk":
            curriculum = Learnability(
                sample_env.task_space,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                sampling="topk",
                topk=args.learnability_topk,
                learnable_prob=args.learnability_prob)
        elif args.curriculum_method == "omni_learnability":
            curriculum = OMNILearnability(
                sample_env.task_space,
                interestingness=interestingness,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                sampling="dist")
        elif args.curriculum_method == "stratified_learnability":
            curriculum = StratifiedLearnability(
                sample_env.task_space,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
                sampling="dist",
                normalize_success=args.normalize_success,
                selection_metric=args.selection_metric
            )
        elif args.curriculum_method == "stratified_dr":
            curriculum = StratifiedDomainRandomization(
                sample_env.task_space,
                eval_envs=syllabus_eval_envs,
                evaluator=evaluator,
                eval_interval_steps=args.eval_interval * args.frames_per_proc * args.procs,
                recurrent_size=acmodel.memory_size,
                recurrent_method="rnn",
                task_names=task_names,
                eval_eps=eval_eps,
                baseline_eval_eps=eval_eps,
            )
        elif args.curriculum_method == "plr":
            curriculum = CentralPrioritizedLevelReplay(
                sample_env.task_space,
                task_names=task_names,
                num_steps=args.frames_per_proc,
                num_processes=args.procs,
                gamma=args.discount,
                gae_lambda=args.gae_lambda,
                task_sampler_kwargs_dict={"strategy": "value_l1",
                                          "staleness_coef": args.plr_staleness_coef,
                                          "temperature": args.plr_temperature},
            )
        elif args.curriculum_method == "sequential":
            curriculum = SequentialCurriculum(
                [["collect_drink", "collect_wood"], ["place_table", "make_wood_pickaxe"], ["make_wood_pickaxe", "collect_stone", "place_table"], [
                    "collect_stone", "place_table", "make_stone_pickaxe", "place_furnace"], ["make_stone_pickaxe", "collect_iron", "place_furnace", "collect_coal"], ["collect_iron", "collect_coal", "place_furnace", "make_iron_pickaxe"], ["make_iron_pickaxe", "collect_diamond"]],
                ["steps>=65536", "steps>=65536", "steps>=98304", "steps>=524288", "steps>=524288", "steps>=524288"],
                sample_env.task_space,
            )
        else:
            raise ValueError("Invalid curriculum method")
        curriculum = make_multiprocessing_curriculum(curriculum, timeout=3000)

    # Load environments
    envs = []
    for i in range(args.procs):
        if args.syllabus:
            envs.append(make_env(curriculum=curriculum)())
        else:
            envs.append(env_module.Env(dummy_bits=args.dummy_bits))
    txt_logger.info("Environments loaded\n")

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, curriculum=curriculum if isinstance(curriculum.curriculum, CentralPrioritizedLevelReplay) else None)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    should_eval = not args.syllabus or (args.syllabus and args.curriculum_method == "plr") or args.full_eval
    print(should_eval)
    if args.eval_interval > 0 and should_eval:
        print("Initial Evaluating")
        rdn_tsr = get_rdn_tsr(eval_envs.envs[0])
        # rdn_tsr = np.zeros(len(eval_envs.given_achievements))
        raw_tsr = status["raw_tsr"]
        ema_tsr = status["ema_tsr"]
        p_fast = status["p_fast"]
        p_slow = status["p_slow"]
        # push saved info to envs
        if p_fast is not None:  # checking one is enough
            p_theta = args.p_theta
            p_fast_reweigh = ((1 - p_theta) * p_fast) / (p_fast + p_theta * (1 - 2 * p_fast))
            p_slow_reweigh = ((1 - p_theta) * p_slow) / (p_slow + p_theta * (1 - 2 * p_slow))
            learning_progress = np.abs(p_fast_reweigh - p_slow_reweigh)
            info = {
                'learning_progress': learning_progress,
                'raw_tsr': raw_tsr,
                'ema_tsr': ema_tsr,
            }
            algo.env.push_info(info)

        task_sampled_rates = np.zeros(len(eval_envs.envs[0].given_achievements))
        # evaluate once before any training
        raw_tsr = eval_all_tasks(acmodel, eval_envs, num_eps=eval_eps)
        # Write task success rates to tensorboard
        header = [f'train_eval/{ach_to_string(ach)}-sr' for ach in envs[0].target_achievements]
        for field, value in zip(header, raw_tsr):
            tb_writer.add_scalar(field, value, num_frames)
        print("Initial Evaluated")

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Save relevant env info
        if args.eval_interval > 0 and should_eval:
            done_envinfo = logs["done_envinfo"]
            for einfo in done_envinfo:
                task_sampled_rates += list(einfo['given_achs'].values())

        # Evaluate for learning progress
        if args.eval_interval > 0 and update % args.eval_interval == 0 and should_eval:
            print("Evaluating")
            raw_tsr = eval_all_tasks(acmodel, eval_envs, num_eps=eval_eps)
            # normalize task success rates with random baseline rates
            norm_tsr = np.maximum(raw_tsr - rdn_tsr, np.zeros(raw_tsr.shape)) / (1.0 - rdn_tsr)
            # exponential mean average learning progress
            ema_tsr = raw_tsr * args.ema_alpha + ema_tsr * (1 - args.ema_alpha) if ema_tsr is not None else raw_tsr
            p_fast = norm_tsr * args.ema_alpha + p_fast * (1 - args.ema_alpha) if p_fast is not None else norm_tsr
            p_slow = p_fast * args.ema_alpha + p_slow * (1 - args.ema_alpha) if p_slow is not None else p_fast
            # NOTE: weighting to give more focus to tasks with lower success probabilities
            p_theta = args.p_theta
            p_fast_reweigh = ((1 - p_theta) * p_fast) / (p_fast + p_theta * (1 - 2 * p_fast))
            p_slow_reweigh = ((1 - p_theta) * p_slow) / (p_slow + p_theta * (1 - 2 * p_slow))
            # learning progress is the change in probability to task success rate
            # NOTE: using bidirectional LP
            learning_progress = np.abs(p_fast_reweigh - p_slow_reweigh)

            # Push information to each environment
            info = {
                'learning_progress': learning_progress,
                'raw_tsr': raw_tsr,
                'ema_tsr': ema_tsr,
            }
            algo.env.push_info(info)
            # Write task success rates to tensorboard
            header = [f'train_eval/{ach_to_string(ach)}-sr' for ach in envs[0].target_achievements]
            for field, value in zip(header, raw_tsr):
                tb_writer.add_scalar(field, value, num_frames)
            # Write saved env info to tensorboard
            sum_tsar = np.sum(task_sampled_rates)
            task_sampled_rates /= sum_tsar if sum_tsar > 0 else 1
            header = [f'train_sampled/{ach_to_string(ach)}' for ach in envs[0].target_achievements]
            header.append('train_sampled/dummy')  # all dummy tasks are the same, so keep track of one
            for field, value in zip(header, task_sampled_rates):
                tb_writer.add_scalar(field, value, num_frames)
            tb_writer.add_scalar("raw_tsr", np.mean(raw_tsr), num_frames)
            tb_writer.add_scalar("ema_tsr", np.mean(ema_tsr), num_frames)
            tb_writer.add_scalar("learning_progress", np.mean(learning_progress), num_frames)

            task_sampled_rates = np.zeros(len(eval_envs.envs[0].given_achievements))
            print("Evaluated")

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            # if status["num_frames"] == 0:
            #     csv_logger.writerow(header)
            # csv_logger.writerow(data)
            # csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

            if args.syllabus:
                curriculum.log_metrics(tb_writer, None, num_frames, log_n_tasks=15)

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0 and not args.syllabus:
            status = {"num_frames": num_frames, "update": update,
                      "p_fast": p_fast, "p_slow": p_slow, "raw_tsr": raw_tsr, "ema_tsr": ema_tsr,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            utils.save_status(status, model_dir, suffix=str(update))
            txt_logger.info("Status saved")
            # Update final checkpoint status
            if update % (args.save_interval * 1) == 0:
                utils.save_status(status, model_dir)
                txt_logger.info("Final status updated")

    # Save final checkpoint status
    if not args.syllabus:
        status = {"num_frames": num_frames, "update": update,
                  "p_fast": p_fast, "p_slow": p_slow, "raw_tsr": raw_tsr, "ema_tsr": ema_tsr,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        utils.save_status(status, model_dir)
        txt_logger.info("Final status saved")
