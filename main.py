import os
import numpy as np
import matplotlib.pyplot as plt

from classes.DeliveryAgent import DeliveryAgent
from classes.DeliveryGrid import DeliveryGrid


def main():
    env = DeliveryGrid()

    # Parâmetros do agente
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01

    # Criar o agente
    agent = DeliveryAgent(env, learning_rate, discount_factor, epsilon, min_epsilon, decay_rate)

    # Configuração a partir de variáveis de ambiente
    episodes = int(os.getenv('EPISODES', '10000'))

    # Treinamento do agente
    print("Iniciando o treinamento para {} episódios...".format(episodes))
    agent.train(episodes)
    print("Treinamento concluído.")

    # plotando
    plt.plot(agent.episode_rewards_list)
    plt.title("Recompensas por Episódio")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa")
    plt.show()

    # Salvar a Q-Table em um arquivo (opcional)
    np.save('q_table.npy', agent.q_table)


if __name__ == "__main__":
    main()
