import random
import numpy as np
from puzzle import GameGrid

class QLearningAgent:
    def __init__(self, grid_len=2, gamma=0.9, alpha=0.5, epsilon=1.0, epsilon_decay=0.99):
        # Параметры Q-Learning
        self.gamma = gamma  # Коэффициент дисконтирования
        self.alpha = alpha  # Скорость обучения
        self.epsilon = epsilon  # Параметр ε для ε-жадной стратегии
        self.epsilon_decay = epsilon_decay
        self.grid_len = grid_len

        # Q-таблица в виде словаря
        self.q_table = {}

        # Игра
        self.game = GameGrid(grid_len=grid_len)

    def get_state(self):
        """
        Преобразует текущее состояние игры в строку, чтобы использовать его в Q-таблице.
        """
        return str(self.game.state()["matrix"])

    def choose_action(self, state):
        """
        Выбирает действие на основе ε-жадной политики.
        """
        actions = ['up', 'down', 'left', 'right']

        # Исследование: случайный выбор действия
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)

        # Эксплуатация: выбор лучшего действия из Q-таблицы
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)

        # Если состояния нет в таблице, выбираем случайное действие
        return random.choice(actions)

    def update_q_value(self, state, action, reward, next_state):
        """
        Обновляет Q-значение с использованием уравнения Q-Learning.
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ['up', 'down', 'left', 'right']}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in ['up', 'down', 'left', 'right']}

        # Формула обновления Q-значения
        max_next_q = max(self.q_table[next_state].values())
        bellman_equation = reward + self.gamma * max_next_q
        td_error = bellman_equation - self.q_table[state][action]

        self.q_table[state][action] += self.alpha * td_error

    def train(self, episodes=1000):
        """
        Тренирует агента с помощью Q-Learning.
        """
        for episode in range(episodes):
            self.game.reset()
            state = self.get_state()
            total_reward = 0

            while not self.game.state()["game_over"]:
                action = self.choose_action(state)
                prev_score = self.game.state()["score"]

                # Выполняем действие
                self.game.move(action)
                next_state = self.get_state()
                reward = self.game.state()["score"] - prev_score

                # Обновляем Q-таблицу
                self.update_q_value(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            # Уменьшаем ε после каждой игры
            self.epsilon *= self.epsilon_decay
            self.alpha = 0.5 - episode * (0.5 - 0.01) / episodes



        # Вывод информации об эпизоде
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {self.epsilon:.4f}, Alpha = {self.alpha:.4f}")

    def play(self):
        """
        Запускает обученную модель для игры.
        """
        self.game.reset()
        while not self.game.state()["game_over"]:
            state = self.get_state()
            action = max(self.q_table.get(state, {}), key=self.q_table.get(state, {}).get, default='up')
            self.game.move(action)
            print(self.game.state())

if __name__ == "__main__":
    # Инициализация агента
    agent = QLearningAgent(grid_len=2)
    agent.train(episodes=1000)  # Тренируем агента на 1000 эпизодов

    # 1. Вывод количества записей в Q-таблице
    print(f"Количество записей в Q-таблице: {len(agent.q_table)}")

    # 2. Проверка жадной стратегии для заданных состояний
    test_states = [
        [[2, 2], [0, 0]],  # Пример 1
        [[2, 0], [2, 0]],  # Пример 2
        [[2, 0], [0, 2]],  # Пример 3
        [[0, 2], [2, 0]],  # Пример 4
        [[0, 0], [2, 2]],  # Пример 5
        [[0, 2], [0, 2]],  # Пример 6
    ]

    actions = ['up', 'down', 'left', 'right']

    print("\nАнализ жадной стратегии:")
    for state in test_states:
        state_str = str(state)  # Преобразуем состояние в строку для Q-таблицы
        print(f"\nСостояние:\n{np.array(state)}")
        if state_str in agent.q_table:
            # Выбор лучшего действия
            best_action = max(agent.q_table[state_str], key=agent.q_table[state_str].get)
            print(f"Лучшее действие: {best_action}")
        else:
            print("Q-таблица не содержит записей для этого состояния.")