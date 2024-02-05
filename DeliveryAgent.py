import numpy as np
import random


class DeliveryAgent:
    def __init__(self, environment):
        self.env = environment
        self.q_table = np.zeros((environment.size, environment.size, 4))  # Inicializa a Q-Table

    def choose_action(self, state):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.epsilon:  # Exploração
            action = np.argmax(self.q_table[state])
        else:  # Exploração
            action = random.randint(0, 3)  # Existem 4 ações possíveis
        return action

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[state][action] = new_q

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset_agent()  # Inicia um novo episódio e obtém o estado inicial
            done = False

            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)  # Realiza a ação e recebe o novo estado e recompensa
                self.update_q_table(state, action, reward, new_state)
                state = new_state

            # Reduzir epsilon (taxa de exploração)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

# Instanciando o agente
# agent = DeliveryAgent(env)
