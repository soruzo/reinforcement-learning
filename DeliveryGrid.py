import numpy as np
import random


class DeliveryGrid:
    def __init__(self, size=10, num_obstacles=5, num_deliveries=3):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.place_objects(num_obstacles, 1)  # Coloca obstáculos
        self.place_objects(num_deliveries, 2)  # Coloca pontos de entrega
        self.reset_agent()  # Posiciona o agente

    def place_objects(self, count, object_type):
        placed = 0
        while placed < count:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = object_type
                placed += 1

    def reset_agent(self):
        while True:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x, y] == 0:
                self.agent_position = (x, y)
                break

    def display(self):
        temp_grid = np.array(self.grid)
        temp_grid[self.agent_position] = 3
        print(temp_grid)


# Exemplo de uso
# env = DeliveryGrid()
# env.display()
