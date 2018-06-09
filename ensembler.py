import numpy as np

import visualizer
import simulator
from environment import Environment
import agent

def ensemble_test(agents: list, test_env):

    print('ensemble test start...')

    test_obs, test_fps = simulator.policy_simulator(test_env, agents[0], do_action=False)
    test_obs, test_fps = test_obs[1:], test_fps[1:]

    print('agent 자기소개 시작')
    for i, pg_agent in enumerate(agents):
        test_reward, test_pv, test_ir, test_pv_vec = pg_agent.run_batch(test_obs[1:], test_fps[1:], is_train=False,
                                                                    verbose=False)
        print(test_pv_vec)
        print("[agent #{} test] reward:{:9.6f} PV:{:9.6f} IR:{:9.6f}".format(i, test_reward, test_pv, test_ir))


    test_env.reset()
    print('자, 이제.. 힘을 합쳐봐!!!')
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
        actions /= 3
        print('MEAN     : ', end='')
        for a in actions:
            print("{:6.2f}".format(a * 100), end=' ')
        observation, pv, done, future_price = test_env.step(actions)
        print(pv, '\n')
