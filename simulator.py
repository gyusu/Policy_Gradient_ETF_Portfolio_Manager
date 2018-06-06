from environment import Environment
from agent import Agent


def policy_simulator(env: Environment, agent: Agent, do_action=True, verbose=True):
    """
    Agent가 Environment 에 대해 한 episode 를 수행한다.
    :return: (observations, actions, rewards, future_prices)
    """
    obs, actions, rewards, fps = [], [], [], []
    i = 1
    done = False
    observation = env.reset()

    if do_action:
        while not done:
            obs.append(observation[0])

            action = agent.decide_action(observation)
            observation, reward, done, future_price = env.step(action[0])

            actions.append(action[0])
            rewards.append(reward)
            fps.append(future_price)

            if verbose:
                print("Day {} Portfolio Value : {}".format(i, reward))

            i += 1
        return obs, actions, rewards, fps

    else:
        while not done:
            obs.append(observation[0])

            observation, done, future_price = env.observe()

            fps.append(future_price)

            i += 1
        return obs, fps