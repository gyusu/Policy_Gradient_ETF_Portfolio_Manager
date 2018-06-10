import numpy as np

import visualizer
import simulator
from environment import Environment
import agent

def ensemble_test(agents: list, val_env, test_env, use_top_n, mkt_pv_vec):

    print('ensemble test start...')

    val_obs, val_fps = simulator.policy_simulator(val_env, agents[0], do_action=False)
    test_obs, test_fps = simulator.policy_simulator(test_env, agents[0], do_action=False)

    print('agent 자기소개 시작')
    for i, pg_agent in enumerate(agents):
        val_reward, val_pv, val_ir, val_pv_vec = pg_agent.run_batch(val_obs, val_fps, is_train=False,
                                                                    verbose=False)
        pg_agent.val_reward = val_reward
        print("[agent #{}  val] reward:{:9.6f} PV:{:9.6f} IR:{:9.6f}".format(i, val_reward, val_pv, val_ir))
        test_reward, test_pv, test_ir, test_pv_vec = pg_agent.run_batch(test_obs, test_fps, is_train=False,
                                                                    verbose=False)
        pg_agent.test_reward = test_reward
        pg_agent.test_pv_vec = np.cumprod(test_pv_vec)
        print("[agent #{} test] reward:{:9.6f} PV:{:9.6f} IR:{:9.6f}".format(i, test_reward, test_pv, test_ir))

    print('validation reward가 높은 상위 {}개 agent만 사용하여 테스트 진행'.format(use_top_n))
    # validation reward 기준 내림차순 정렬
    agents.sort(key=lambda a: a.val_reward, reverse=True)
    for agent in agents:
        print(agent.name, agent.val_reward, agent.test_reward)
    agents = agents[:use_top_n]

    test_env.reset()
    ensemble_pv_vec = []
    for obs in test_obs:
        actions = []
        for i, pg_agent in enumerate(agents):
            act = pg_agent.decide_action([obs])
            print('agent #{:02}: '.format(i), end='')
            for a in act[0]:
                print("{:6.2f}".format(a * 100), end=' ')
            print()
            actions.append(act[0])
        actions = np.array(actions).cumsum(axis=0)[-1]
        actions /= len(agents)
        print('AVG      : ', end='')
        for a in actions:
            print("{:6.2f}".format(a * 100), end=' ')
        observation, pv, done, future_price = test_env.step(actions)
        print(pv, '\n')

        ensemble_pv_vec.append(pv)

    agent_pv_vec_list = []
    for agent in agents:
        agent_pv_vec_list.append(agent.test_pv_vec)
    visualizer.plot_ensemble_pv(agent_pv_vec_list, ensemble_pv_vec, mkt_pv_vec)