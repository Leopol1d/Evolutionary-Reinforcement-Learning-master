import pygame
import torch as th
from torch import nn
import configparser
from torch.optim import Adam, RMSprop
from torch.utils.tensorboard import SummaryWriter

config_dir = 'configs/configs_dqn.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True # 每次返回的卷积算法将是确定的

import os, logging
import numpy as np
from single_agent.Model_common import ActorNetwork, CriticNetwork
from single_agent.utils_common import identity, to_tensor_var
from single_agent.Memory_common import OnPolicyReplayMemory

import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.my_runner_ppo import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from single_agent.utils_common import agg_double_list
import matplotlib.pyplot as plt

import torch
from copy import deepcopy
from EA import EA

class My_AL:
    """
    An multi-agent learned with DQN
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R", pop_size=10):
        self.pop_size = pop_size
        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]
        self.reward_type = reward_type
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state, self.action_mask = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.test_seeds = test_seeds
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        self.memory = OnPolicyReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.n_agents = len(self.env.controlled_vehicles)
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]


        self.writer = SummaryWriter(log_dir='Results/tensorboard/')
        self.savefolder = 'Results/Plots/'
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)

        self.manager = Manager()
        # Evolution
        self.evolver = EA()

        self.population = self.manager.list()
        self.pop_optimizer = []
        for i in range(self.pop_size):
            self.population.append(ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))
            if self.optimizer_type == "adam":
                self.pop_optimizer.append(Adam(self.population[i].parameters(), lr=self.actor_lr))
            elif self.optimizer_type == "rmsprop":
                self.pop_optimizer.append(RMSprop(self.population[i].parameters(), lr=self.actor_lr))

        self.best_policy = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)

        # self.learner = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)


        # Initialize Rollout Bucket
        self.rollout_bucket = self.manager.list()
        for _ in range(5):
            self.rollout_bucket.append(ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        ############## MULTIPROCESSING TOOLS ###################
        # Evolutionary population Rollout workers
        self.evo_task_pipes = [Pipe() for _ in range(self.pop_size)]
        self.evo_result_pipes = [Pipe() for _ in range(self.pop_size)]
        self.evo_workers = [Process(target=rollout_worker, args=(
        id, self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], True, self.population, self.critic, self.memory, self.env, self.reward_scale, self.reward_type, self.reward_gamma))
                            for id in range(self.pop_size)]
        for worker in self.evo_workers: worker.start()
        self.evo_flag = [True for _ in range(self.pop_size)]

        # Learner rollout workers
        self.task_pipes = [Pipe() for _ in range(5)]
        self.result_pipes = [Pipe() for _ in range(5)]
        self.workers = [Process(target=rollout_worker, args=(
        id, self.task_pipes[id][1], self.result_pipes[id][0], True, self.population, self.critic, self.memory, self.env, self.reward_scale, self.reward_type, self.reward_gamma))
                        for id in range(5)]
        for worker in self.workers: worker.start()
        self.roll_flag = [True for _ in range(5)]

        # Test bucket
        self.test_bucket = self.manager.list()
        self.test_bucket.append(ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        # Test workers
        self.test_task_pipes = [Pipe() for _ in range(3)]
        self.test_result_pipes = [Pipe() for _ in range(3)]
        self.test_workers = [Process(target=rollout_worker, args=(
        id, self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.population, self.critic, self.memory, self.env, self.reward_scale, self.reward_type, self.reward_gamma))
                             for id in range(3)]
        for worker in self.test_workers: worker.start()
        self.test_flag = False

        # Trackers
        self.best_score = -float('inf');
        self.gen_frames = 0;
        self.total_frames = 0;
        self.test_score = None;
        self.test_std = None


    def forward_generation(self, gen):
        gen_max = -float('inf')

        # Start Evolution rollouts

        for id, actor in enumerate(self.population):
            self.evo_task_pipes[id][0].send(id)

        # Sync all learners actor to cpu (rollout) actor and start their rollout
        self.actor.cpu()  # learner的q_network
        for rollout_id in range(len(self.rollout_bucket)):
            utils.hard_update(self.rollout_bucket[rollout_id], self.actor)
            self.task_pipes[rollout_id][0].send(0)
        self.actor.to(device=self.device)

        # Start Test rollouts
        if gen % 2 == 0:
            self.test_flag = True
            for pipe in self.test_task_pipes: pipe[0].send(0)

        ############# UPDATE PARAMS USING GRADIENT DESCENT ##########

        if self.memory.__len__() >= self.episodes_before_train:
            for _ in range(10):
                self.bp(self.actor, self.actor_optimizer, update_target=True)
            for i in range(self.pop_size):
                self.bp(self.population[i], self.pop_optimizer[i], update_target=False)

        ########## JOIN ROLLOUTS FOR EVO POPULATION ############
        all_fitness = []

        for i in range(self.pop_size):
            _, fitness, _ = self.evo_result_pipes[i][1].recv()

            all_fitness.append(fitness)


            self.best_score = max(self.best_score, fitness)
            gen_max = max(gen_max, fitness)

        ########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
        rollout_fitness = [];

        for i in range(5):
            _, fitness, _ = self.result_pipes[i][1].recv()
            self.best_score = max(self.best_score, fitness)
            gen_max = max(gen_max, fitness)
            rollout_fitness.append(fitness);


        ######################### END OF PARALLEL ROLLOUTS ################

        ############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
        champ_index = all_fitness.index(max(all_fitness))
        utils.hard_update(self.test_bucket[0], self.population[champ_index])
        if max(all_fitness) > self.best_score:
            self.best_score = max(all_fitness)
            utils.hard_update(self.best_policy, self.population[champ_index])
            torch.save(self.population[champ_index].state_dict(), '_best')
            print("Best policy saved with score", '%.2f' % max(all_fitness))


        ###### TEST SCORE ######
        if self.test_flag:
            self.test_flag = False
            test_scores = []
            for pipe in self.test_result_pipes:  # Collect all results
                _, fitness, _ = pipe[1].recv()
                self.best_score = max(self.best_score, fitness)
                gen_max = max(gen_max, fitness)
                test_scores.append(fitness)
            test_scores = np.array(test_scores)
            test_mean = np.mean(test_scores);
            test_std = (np.std(test_scores))

        else:
            test_mean, test_std = None, None

        # NeuroEvolution's probabilistic selection and recombination step

        self.evolver.epoch(gen, self.population, all_fitness, self.rollout_bucket)

        # Compute the champion's eplen

        return gen_max, test_mean, test_std, rollout_fitness



    def train(self, Max_EPISODES):
        # Define Tracker class to track scores
        eval_rewards = []
        for gen in range(1, Max_EPISODES + 1):

            # Train one iteration
            max_fitness,test_mean, test_std, rollout_fitness = self.forward_generation(gen)
            if test_mean: self.writer.add_scalar('test_score', test_mean, gen)

            print('Gen/Frames:', gen,
                  ' Gen_max_score:', '%.2f' % max_fitness,
                 'Test_score u/std', utils.pprint(test_mean),
                  utils.pprint(test_std))

            if gen % 5 == 0:
                rewards, _ = self.evaluation()
                rewards_mu, rewards_std = agg_double_list(rewards)
                print("Gen %d, Average Reward %.2f" % (gen, rewards_mu))
                print('Best_score_ever:''/', '%.2f' % self.best_score)
                print()
                # episodes.append(madqn.n_episodes + 1)
                eval_rewards.append(rewards_mu)

        plt.figure()

        eval_rewards = np.array(eval_rewards)
        plt.plot(eval_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend(["EMAPPO"])

        plt.show()
        np.save(r'D:\Software\work\PyCharm\Workspace\Debug\Evolutionary-Reinforcement-Learning-master'
                r'\Results\eval_reward\eval_rewards_emappo.npy', eval_rewards)

        ###Kill all processes
        try:
            for p in self.task_pipes: p[0].send('TERMINATE')
            for p in self.test_task_pipes: p[0].send('TERMINATE')
            for p in self.evo_task_pipes: p[0].send('TERMINATE')
        except:
            None

    def evaluation(self, eval_episodes=3):
        # self.env.close() # 先把训练env关闭，不然打开测试env会报错(只允许一个env存在)
        rewards = []
        infos = []
        for i in range(eval_episodes):
            rewards_i = []
            infos_i = []
            state, _ = self.env.reset()
            state = torch.FloatTensor(state)
            done = False
            while not done:
                # self.env.render()
                V = self.best_policy(state)
                V = V.data.numpy()
                action = np.argmax(V, axis=1)
                state, reward, done, info = self.env.step(action)
                state = torch.FloatTensor(state)
                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward)
                infos_i.append(info)
            rewards.append(rewards_i)
            infos.append(infos_i)
        # self.env.close()
        return rewards, infos

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def bp(self, actor, actor_optimizer, update_target=False):
        self.n_episodes += 1

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        for agent_id in range(self.n_agents):
            # update actor network
            actor_optimizer.zero_grad()
            values = self.critic_target(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()
            advantages = rewards_var[:, agent_id, :] - values

            action_log_probs = actor(states_var[:, agent_id, :])  # [100, 5]
            # 被选中动作的值
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
            # target_net中选中动作的值
            old_action_log_probs = self.actor_target(states_var[:, agent_id, :]).detach()
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:, agent_id, :], 1)
            ratio = th.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -th.mean(th.min(surr1, surr2))
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            actor_optimizer.step()

            # update critic network
            self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        if update_target:
            # update actor target network and critic target network
            if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
                self._soft_update_target(self.actor_target, actor)
                self._soft_update_target(self.critic_target, self.critic)