import numpy

from core import utils as utils
import numpy as np
import torch
import gym
from algos.single_agent.Memory_common import ReplayMemory

from algos.single_agent.utils_common import identity, to_tensor_var, index_to_one_hot
# Rollout evaluate an agent in a complete game

@torch.no_grad()
def rollout_worker(id, task_pipe, result_pipe, store_data, model_bucket,
                   critic, memory, env, reward_scale, reward_type, reward_gamma):

    n_agents = len(env.controlled_vehicles)

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
        state, _ = env.reset()

        states = []
        actions = []
        rewards = []
        done = True
        average_speed = 0

        while True:  # unless done
            states.append(state)
            action = explore_action(state, n_agents, net)
            next_state, global_reward, done, info = env.step(tuple(action))
            actions.append([index_to_one_hot(a, env.n_a) for a in action])
            if reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif reward_type == "global_R":
                reward = [global_reward] * n_agents # [0.81223640832933, 0.81223640832933, 0.81223640832933, 0.81223640832933]
            rewards.append(reward)
            fitness += global_reward

            average_speed += info["average_speed"]
            # next_state = torch.FloatTensor(next_state)
            final_state = next_state
            state = next_state

            if done:
                state, _ = env.reset()
                break

        if done:
            final_value = [0.0] * n_agents
        else:
            final_action = clean_action(state, n_agents, net)
            final_value = value(final_state, final_action, env, n_agents, critic)

        if reward_scale > 0:
            rewards = np.array(rewards) / reward_scale

        if store_data:
            for agent_id in range(n_agents):
                rewards[:, agent_id] = _discount_reward(rewards[:, agent_id], final_value[agent_id], reward_gamma)

        rewards = rewards.tolist()
        memory.push(states, actions, rewards)





        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame])




def explore_action(state, n_agents, net):
    softmax_actions = _softmax_action(state, n_agents, net)
    actions = []
    for pi in softmax_actions:
        actions.append(np.random.choice(np.arange(len(pi)), p=pi))
    return actions

def clean_action(state, n_agents, net):
    softmax_actions = _softmax_action(state, n_agents, net)
    actions = []
    for pi in softmax_actions:
        actions.append(np.random.choice(np.arange(len(pi)), p=pi))
    return actions

def _softmax_action(state, n_agents, net): # state: tensor[[]]
    # state_var = numpy.array([state])
    state_var = to_tensor_var([state], False)
    # state_var = torch.from_numpy(state_var)

    softmax_action = []
    for agent_id in range(n_agents):

        softmax_action_var = torch.exp(net(state_var[:, agent_id, :]))
        # print('softmax_action_var: ', softmax_action_var)
        # print('softmax_action_var.data.numpy(): ', softmax_action_var.data.numpy()[0])
        softmax_action.append(softmax_action_var.data.numpy()[0])

    return softmax_action

def value(state, action, env, n_agents, critic):
    state_var = torch.FloatTensor(state)
    action = index_to_one_hot(action, env.n_a)
    action_var = torch.FloatTensor(action)
    values = [0] * n_agents
    for agent_id in range(n_agents):
        value_var = critic(state_var[:, agent_id, :], action_var[:, agent_id, :])
        values[agent_id] = value_var.data.numpy()
    return values

def _discount_reward(rewards, final_value, reward_gamma):
    discounted_r = np.zeros_like(rewards)
    running_add = final_value
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * reward_gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r