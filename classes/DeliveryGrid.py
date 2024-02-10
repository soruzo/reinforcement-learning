import numpy as np
import random


class DeliveryGrid:
    def __init__(self, size=10, num_obstacles=5, num_deliveries=3):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.place_objects(num_obstacles, 1)
        self.place_objects(num_deliveries, 2)
        self.reset_agent()
        self.delivery_reward = 25
        self.obstacle_penalty = -5
        self.step_penalty = -1
        self.deliveries_made = 3
        self.max_steps = size
        self.num_obstacles = num_obstacles
        self.num_deliveries = num_deliveries
        self.agent_position = []

    def step(self, action):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = directions[action]
        new_position = (self.agent_position[0] + move[0], self.agent_position[1] + move[1])

        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
            if self.is_obstacle(new_position):
                reward = self.obstacle_penalty
                done = False
            elif self.is_delivery_point(new_position):
                reward = self.delivery_reward
                self.grid[new_position] = 0
                self.deliveries_made += 1
                done = self.check_all_deliveries_made()
            else:
                reward = self.step_penalty
                done = False
            if not self.is_obstacle(new_position):
                self.agent_position = new_position
        else:
            reward = self.obstacle_penalty
            done = False

        self.max_steps -= 1
        if self.max_steps <= 0:
            done = True

        new_state = self.agent_position

        return new_state, reward, done

    def is_obstacle(self, position):
        x, y = position
        return self.grid[x][y] == 1

    def is_delivery_point(self, position):
        x, y = position
        return self.grid[x][y] == 2

    def check_all_deliveries_made(self):
        return self.deliveries_made == self.num_deliveries

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.place_objects(self.num_obstacles, 1)
        self.place_objects(self.num_deliveries, 2)
        self.reset_agent()
        self.deliveries_made = 0
        self.max_steps = self.size
        return self.agent_position

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
