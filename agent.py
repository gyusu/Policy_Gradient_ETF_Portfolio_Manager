from environment import Environment

class Agent:

    def __init__(self, env:Environment):
        self.env = env

    def decide_action(self, observation):
        return self.env.action_sample()
