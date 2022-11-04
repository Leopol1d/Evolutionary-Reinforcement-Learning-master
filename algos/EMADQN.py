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
from single_agent.Model_common import ActorNetwork
from single_agent.utils_common import identity, to_tensor_var
from single_agent.Memory_common import ReplayMemory

import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.my_runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from single_agent.utils_common import agg_double_list
import matplotlib.pyplot as plt

import torch

from EA import EA

class My_AL:
    """
    An multi-agent learned with DQN
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=20.,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True, target_update_freq=4, reward_type="regionalR"):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale


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

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_cuda = use_cuda and th.cuda.is_available()
        self.q_network = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                      self.action_dim, self.actor_output_act).to(device=self.device)
        self.target_network = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                           self.action_dim, self.actor_output_act).to(device=self.device)
        self.target_update_freq = target_update_freq
        self.reward_type = reward_type
        self.episode_rewards = [0]

        if self.optimizer_type == "adam":
            self.q_netwok_optimizer = Adam(self.q_network.parameters(), lr=self.actor_lr)
        elif self.optimizer_type == "rmsprop":
            self.q_netwok_optimizer = RMSprop(self.q_network.parameters(), lr=self.actor_lr)
        if self.use_cuda:
            self.q_network.cuda()
            self.target_network.cuda()


        self.writer = SummaryWriter(log_dir='Results/tensorboard/')
        self.savefolder = 'Results/Plots/'
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)

        self.manager = Manager()
        # Evolution
        self.evolver = EA()

        self.population = self.manager.list()
        for _ in range(10):
            self.population.append(ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        self.best_policy = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)

        # self.learner = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)

        self.memory = ReplayMemory(memory_capacity)

        # Initialize Rollout Bucket
        self.rollout_bucket = self.manager.list()
        for _ in range(5):
            self.rollout_bucket.append(ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        ############## MULTIPROCESSING TOOLS ###################
        # Evolutionary population Rollout workers
        self.evo_task_pipes = [Pipe() for _ in range(10)]
        self.evo_result_pipes = [Pipe() for _ in range(10)]
        self.evo_workers = [Process(target=rollout_worker, args=(
        id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], True, self.population, self.memory, self.env))
                            for id in range(10)]
        for worker in self.evo_workers: worker.start()
        self.evo_flag = [True for _ in range(10)]

        # Learner rollout workers
        self.task_pipes = [Pipe() for _ in range(5)]
        self.result_pipes = [Pipe() for _ in range(5)]
        self.workers = [Process(target=rollout_worker, args=(
        id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], True, self.rollout_bucket, self.memory, self.env))
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
        id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.test_bucket, self.memory, self.env)) for id in
                             range(3)]
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
        self.q_network.cpu()  # learner的q_network
        for rollout_id in range(len(self.rollout_bucket)):
            utils.hard_update(self.rollout_bucket[rollout_id], self.q_network)
            self.task_pipes[rollout_id][0].send(0)
        self.q_network.to(device=self.device)

        # Start Test rollouts
        if gen % 2 == 0:
            self.test_flag = True
            for pipe in self.test_task_pipes: pipe[0].send(0)

        ############# UPDATE PARAMS USING GRADIENT DESCENT ##########

        if self.memory.__len__() >= self.episodes_before_train:
            for _ in range(10):
                batch = self.memory.sample(self.batch_size)
                states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
                actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
                rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
                next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
                dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)
                # compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                current_q = self.q_network(states_var).gather(1, actions_var)
                # compute V(s_{t+1}) for all next states and all actions,
                # and we then take max_a { V(s_{t+1}) }
                next_state_action_values = self.target_network(next_states_var).detach()
                next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
                # compute target q by: r + gamma * max_a { V(s_{t+1}) }
                target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

                # update value network
                self.q_netwok_optimizer.zero_grad()
                if self.critic_loss == "huber":
                    loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
                else:
                    loss = th.nn.MSELoss()(current_q, target_q)
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm(self.q_network.parameters(), self.max_grad_norm)
                self.q_netwok_optimizer.step()

            # Periodically update the target network by Q network to target Q network
            if self.n_episodes % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        ########## JOIN ROLLOUTS FOR EVO POPULATION ############
        all_fitness = [];

        for i in range(10):
            _, fitness, _ = self.evo_result_pipes[i][1].recv()

            all_fitness.append(fitness);


            self.best_score = max(self.best_score, fitness)
            gen_max = max(gen_max, fitness)

        ########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
        rollout_fitness = [];

        for i in range(5):
            _, fitness, pg_frames = self.result_pipes[i][1].recv()
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
        for gen in range(1, Max_EPISODES + 1):  # Infinite generations

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
                print("Gem %d, Average Reward %.2f" % (gen, rewards_mu))
                print('Best_score_ever:''/', '%.2f' % self.best_score)
                print()
                # episodes.append(madqn.n_episodes + 1)
                eval_rewards.append(rewards_mu)

        plt.figure()
        plt.plot(eval_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend(["DQN"])
        plt.show()

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
                self.env.render()
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
        self.env.close()
        return rewards, infos

    def bp(self):
        self.n_episodes += 1

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)
        # compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_q = self.q_network(states_var).gather(1, actions_var)
        # compute V(s_{t+1}) for all next states and all actions,
        # and we then take max_a { V(s_{t+1}) }
        next_state_action_values = self.target_network(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        # compute target q by: r + gamma * max_a { V(s_{t+1}) }
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

        # update value network
        self.q_netwok_optimizer.zero_grad()
        if self.critic_loss == "huber":
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            loss = th.nn.MSELoss()(current_q, target_q)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.q_network.parameters(), self.max_grad_norm)
        self.q_netwok_optimizer.step()

        # Periodically update the target network by Q network to target Q network

    # if self.n_episodes % self.target_update_freq == 0:
    #     self.target_network.load_state_dict(self.q_network.state_dict())