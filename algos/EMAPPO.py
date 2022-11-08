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
th.backends.cudnn.deterministic = True  # 每次返回的卷积算法将是确定的

import os, logging
import numpy as np
from single_agent.Model_common import ActorNetwork, CriticNetwork
from single_agent.utils_common import identity, to_tensor_var
from single_agent.Memory_common import OnPolicyReplayMemory

import sys

sys.path.append("..")
from single_agent.utils import agg_double_list, VideoRecorder

import numpy as np, os, time, random, torch, sys
from neuroevolution import SSNE
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

    def __init__(self, env, env_eval,state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=2, reward_type="global_R",
                 pop_size=50, rollout_size=10, dirs=None):
        self.dirs = dirs
        self.pop_size = pop_size
        self.rollout_size = rollout_size
        self.exceed_times = 0

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]
        self.reward_type = reward_type
        self.env = env
        self.env_eval = env_eval
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
        self.eval_rewards = [-50]
        self.actor_rewards = [0]
        self.eval_best_policy_flag = False
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
        for _ in range(self.pop_size):
            self.population.append(
                ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        self.best_policy = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)

        # self.learner = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)

        # Initialize Rollout Bucket
        self.rollout_bucket = self.manager.list()
        for _ in range(self.rollout_size):
            self.rollout_bucket.append(
                ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        ############## MULTIPROCESSING TOOLS ###################
        # Evolutionary population Rollout workers
        self.evo_task_pipes = [Pipe() for _ in range(self.pop_size)]
        self.evo_result_pipes = [Pipe() for _ in range(self.pop_size)]
        self.evo_workers = [Process(target=rollout_worker, args=(
            id, self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], True, self.population, self.critic, self.env,
            self.reward_scale, self.reward_type, self.reward_gamma))
                            for id in range(self.pop_size)]
        for worker in self.evo_workers: worker.start()
        self.evo_flag = [True for _ in range(self.pop_size)]

        # Learner rollout workers
        self.task_pipes = [Pipe() for _ in range(self.rollout_size)]
        self.result_pipes = [Pipe() for _ in range(self.rollout_size)]
        self.workers = [Process(target=rollout_worker, args=(
            id, self.task_pipes[id][1], self.result_pipes[id][0], True, self.population, self.critic, self.env,
            self.reward_scale, self.reward_type, self.reward_gamma))
                        for id in range(self.rollout_size)]
        for worker in self.workers: worker.start()
        self.roll_flag = [True for _ in range(self.rollout_size)]

        # Test bucket
        self.test_bucket = self.manager.list()
        self.test_bucket.append(
            ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act))

        # Test workers
        self.test_task_pipes = [Pipe() for _ in range(3)]
        self.test_result_pipes = [Pipe() for _ in range(3)]
        self.test_workers = [Process(target=rollout_worker, args=(
            id, self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.population, self.critic,
            self.env, self.reward_scale, self.reward_type, self.reward_gamma))
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
        applicant = None
        applicant_fitness = -float('inf')
        # Start Evolution rollouts
        # while self.memory.__len__() < self.episodes_before_train:
        for id, actor in enumerate(self.population):
            self.evo_task_pipes[id][0].send(id)

        # Sync all learners actor to cpu (rollout) actor and start their rollout
        self.actor.cpu()  # learner的q_network
        for rollout_id in range(len(self.rollout_bucket)):
            utils.hard_update(self.rollout_bucket[rollout_id], self.actor)
            self.task_pipes[rollout_id][0].send(0)
        self.actor.to(device=self.device)
        ########## JOIN ROLLOUTS FOR EVO POPULATION ############
        all_fitness = []

        for i in range(self.pop_size):
            _, fitness, _, states, actions, rewards = self.evo_result_pipes[i][1].recv()
            self.memory.push(states, actions, rewards)

            all_fitness.append(fitness)

            # self.best_score = max(self.best_score, fitness)
            gen_max = max(gen_max, fitness)

        ########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
        rollout_fitness = []

        for i in range(self.rollout_size):
            _, fitness, _, states, actions, rewards = self.result_pipes[i][1].recv()
            self.memory.push(states, actions, rewards)
            # self.best_score = max(self.best_score, fitness)
            gen_max = max(gen_max, fitness)
            rollout_fitness.append(fitness)

        # Start Test rollouts
        if gen % 10 == 0:
            self.test_flag = True
            for pipe in self.test_task_pipes: pipe[0].send(0)

        ############# UPDATE PARAMS USING GRADIENT DESCENT ##########
        print()
        # print('training...... pop size:', self.pop_size)
        for _ in range(10):
            self.bp(self.actor, self.actor_optimizer, update_target=True)

        ######################### END OF PARALLEL ROLLOUTS ################

        ############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
        champ_index = all_fitness.index(max(all_fitness))

        utils.hard_update(self.test_bucket[0], self.population[champ_index])
        print('Gen: %d, Interacted max_fitness: %.2f, best_score: %.2f' % (gen, max(all_fitness), self.best_score))
        max_pop_fitness = max(all_fitness)
        if max_pop_fitness > self.best_score:
            # utils.hard_update(self.best_policy, self.population[champ_index])
            # torch.save(self.population[champ_index].state_dict(), '_best')
            print('Elitist exceed the best policy!')

        ###### TEST SCORE ######
        if self.test_flag:
            self.test_flag = False
            test_scores = []
            for pipe in self.test_result_pipes:  # Collect all results
                _, fitness, _, states, actions, rewards = pipe[1].recv()
                if fitness > self.best_score:
                    # utils.hard_update(target=self.best_policy, source=self.test_bucket[0])
                    print('test_bucket exceed the best policy!')
                gen_max = max(gen_max, fitness)
                test_scores.append(fitness)
            test_scores = np.array(test_scores)
            test_mean = np.mean(test_scores);
            test_std = (np.std(test_scores))
            max_test_fitness = max(test_scores)


        else:
            test_mean, test_std = None, None
            max_test_fitness = -float('inf')

        if max_pop_fitness > max_test_fitness:
            applicant = self.population[champ_index]
            applicant_fitness = max_pop_fitness
        else:
            applicant = self.test_bucket[0]
            applicant_fitness = max_test_fitness

        if applicant_fitness > self.best_score / 5:
            self.eval_best_policy_flag =True
        else:
            applicant = None

        # NeuroEvolution's probabilistic selection and recombination step

        self.evolver.epoch(gen, self.population, all_fitness, self.rollout_bucket, rollout_fitness, self.best_policy)

        # Compute the champion's eplen

        return gen_max, test_mean, test_std, rollout_fitness, applicant, applicant_fitness

    def train(self, Max_EPISODES):
        # Define Tracker class to track scores

        for gen in range(1, Max_EPISODES + 1):

            max_fitness, test_mean, test_std, rollout_fitness, applicant, applicant_fitness = self.forward_generation(gen)
            if test_mean: self.writer.add_scalar('test_score', test_mean, gen)

            # print('Gen/Frames:', gen,
            #       ' Gen_max_score:', '%.2f' % max_fitness,
            #       'Test_score u/std', utils.pprint(test_mean),
            #       utils.pprint(test_std))

            if gen % 1 == 0:

                if self.eval_best_policy_flag:
                    self.eval_best_policy_flag = False
                    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds = self.evaluation(
                        self.dirs['train_videos'], applicant, self.env_eval, eval_episodes=3,
                        is_train=True)
                    rewards_mu, rewards_std = agg_double_list(rewards)
                    print("Gen %d, Applicant Average Reward %.2f" % (gen, rewards_mu))

                    if rewards_mu > self.eval_rewards[-1]:
                        self.exceed_times += 1
                        utils.hard_update(self.best_policy, applicant)
                        self.eval_rewards[-1] = rewards_mu
                        self.best_score = applicant_fitness
                        print("Break Through! Best policy saved with reward", '%.2f' % rewards_mu)
                    else:
                        rewards_mu = self.eval_rewards[-1]

                else:
                    rewards_mu = self.eval_rewards[-1]
                self.eval_rewards.append(rewards_mu)
                print('Best applicant reward ever: %.2f, exceed times: %d' % (self.eval_rewards[-1], self.exceed_times))

                self.actor.cpu()  # learner的q_network
                rewards, (vehicle_speed, vehicle_position), steps, avg_speeds = self.evaluation(
                    self.dirs['train_videos'], self.actor, self.env_eval, eval_episodes=3, is_train=True)
                rewards_mu_actor, rewards_std = agg_double_list(rewards)
                print("Gen %d, Actor Average Reward %.2f" % (gen, rewards_mu_actor))
                if rewards_mu_actor > self.eval_rewards[-1]:
                    print('actor exceed the best policy!')
                    utils.hard_update(target=self.best_policy, source=self.actor)
                    self.eval_rewards[-1] = rewards_mu_actor
                else:
                    self._soft_update_target(self.actor, self.best_policy)
                # episodes.append(madqn.n_episodes + 1)
                self.actor_rewards.append(rewards_mu_actor)
                self.actor.to(device=self.device)

        self.save(self.dirs['models'], Max_EPISODES)
        plt.figure()
        eval_rewards = np.array(self.eval_rewards)
        actor_rewards = np.array(self.actor_rewards)
        plt.plot(eval_rewards)
        plt.plot(actor_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend(labels=['best_policy', 'learner'], loc='best')
        plt.show()
        np.save(self.dirs['eval_logs'] + 'best_policy_rewards.npy', eval_rewards)
        np.save(self.dirs['eval_logs'] + 'learner_rewards.npy', actor_rewards)

        # dir = 'Results/Nov_05_08_08_49/eval_logs/reward.npy'
        # b = np.load(dir)
        ###Kill all processes
        try:
            for p in self.task_pipes: p[0].send('TERMINATE')
            for p in self.test_task_pipes: p[0].send('TERMINATE')
            for p in self.evo_task_pipes: p[0].send('TERMINATE')
        except:
            None

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def bp(self, actor, actor_optimizer, update_target=False):
        self.n_episodes += 1
        # print('actor == self.actor: ', actor == self.actor)
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

    def evaluation(self, output_dir, policy, env, eval_episodes=3, is_train=True):
        rewards = []
        infos = []
        avg_speeds = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        seeds = [int(s) for s in self.test_seeds.split(',')]

        video_recorder = None

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i+1], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i+1], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i+1], num_CAV=i + 4)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i+1])

            n_agents = len(env.controlled_vehicles)
            # rendered_frame = env.render(mode="rgb_array")
            # if policy == self.best_policy:
            #     video_filename = os.path.join(output_dir,
            #                                   "best_policy_testing_episode{}".format(
            #                                       self.n_episodes + 1) + '_{}'.format(i) +
            #                                   '.mp4')
            # else:
            #     video_filename = os.path.join(output_dir,
            #                                   "learner_testing_episode{}".format(
            #                                       self.n_episodes + 1) + '_{}'.format(i) +
            #                                   '.mp4')

            # if video_filename is not None:
            #     # print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, 5))
            #     video_recorder = VideoRecorder(video_filename,
            #                                    frame_size=rendered_frame.shape, fps=5)
            #     video_recorder.add_frame(rendered_frame)
            # else:
            #     video_recorder = None

            while not done:
                step += 1
                action = self.action(state, n_agents, policy)
                state, reward, done, info = env.step(action)
                avg_speed += info["average_speed"]
                # rendered_frame = env.render(mode="rgb_array")
                # if video_recorder is not None:
                #     video_recorder.add_frame(rendered_frame)
                rewards_i.append(reward)
                infos_i.append(info)

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            #
            # if video_recorder is not None:
            #     video_recorder.release()
            # env.close()
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds

    def action(self, state, n_agents, net):
        softmax_actions = self._softmax_action(state, n_agents, net)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    def _softmax_action(self, state, n_agents, net):  # state: tensor[[]]
        state_var = to_tensor_var([state], False)

        softmax_action = []
        for agent_id in range(n_agents):
            softmax_action_var = torch.exp(net(state_var[:, agent_id, :]))
            # print('softmax_action_var: ', softmax_action_var)
            # print('softmax_action_var.data.numpy(): ', softmax_action_var.data.numpy()[0])
            softmax_action.append(softmax_action_var.data.numpy()[0])

        return softmax_action

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.actor.train()
            else:
                self.actor.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                 'model_state_dict': self.actor.state_dict(),
                 'optimizer_state_dict': self.actor_optimizer.state_dict(),
                 'best_policy': self.best_policy.state_dict()},
                file_path)
