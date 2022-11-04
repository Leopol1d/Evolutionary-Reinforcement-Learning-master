from core import utils as utils
import numpy as np
import torch
import gym
from algos.single_agent.Memory_common import ReplayMemory

from algos.single_agent.utils_common import identity, to_tensor_var

# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, memory, env):

    n_agent = len(env.controlled_vehicles)

    # env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different

    ###LOOP###
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]

        fitness = 0.0
        total_frame = 0
        # print(env.reset())
        state, _ = env.reset()

        while True:  # unless done

            actions = [0] * n_agent

            if type == 'pg':
                for agent_id in range(n_agent):
                    actions[agent_id] = np.random.choice(env.n_a)
            else:
                # state = to_tensor_var([state], False)
                state = torch.FloatTensor(state)
                V = net(state)
                V = V.data.numpy()
                actions = np.argmax(V, axis=1)

            next_state, global_reward, done, info = env.step(tuple(actions))
            reward = list(info["regional_rewards"])
            fitness += sum(reward) / n_agent
            # reward = global_reward
            # fitness += global_reward
            if store_data:
                # for agent_id in range(n_agent):
                #     rollout_trajectory.append(state[agent_id,:], actions[agent_id], reward[agent_id],
                #                               next_state[agent_id], done)
                for agent_id in range(n_agent):
                    memory.push(state[agent_id, :], actions[agent_id], reward[agent_id], next_state[agent_id, :],
                                     done)
            # next_state = to_tensor_var([next_state], False)
            next_state = torch.FloatTensor(next_state)

            state = next_state

            total_frame += 1

            # DONE FLAG IS Received
            if done:

                break

        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame])
