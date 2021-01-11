import pygame
from enum import Enum
from collections import namedtuple
import random
import numpy as np


pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Block = namedtuple("Block", "x, y")

WIN_WIDTH = 640
WIN_HEIGHT = 480
BLOCK_SIZE = 20
GAME_SPEED = 100

SNAKE_HEAD_COLOR = (0, 153, 51)
SNAKE_COLOR = (51, 204, 51)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class Snake:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.head = Block(self.x, self.y)
        self.snake_elements = [
            self.head,
            Block(self.head.x - BLOCK_SIZE, self.head.y),
            Block(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.current_direction = Direction.RIGHT

    def update(self, action):
        clock_wise_direction = [Direction.RIGHT,
                                Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise_direction.index(self.current_direction)

        if np.array_equal(action, [1, 0, 0]):
            # don't change the direction
            new_direction = clock_wise_direction[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_direction = clock_wise_direction[new_idx]
        else:
            new_idx = (idx - 1) % 4
            new_direction = clock_wise_direction[new_idx]

        self.current_direction = new_direction

        # if the snake doesn't eat the apple, pop the tail and insert a new head
        new_head_x = self.head.x
        new_head_y = self.head.y
        if self.current_direction == Direction.RIGHT:
            new_head_x += BLOCK_SIZE
        elif self.current_direction == Direction.LEFT:
            new_head_x -= BLOCK_SIZE
        elif self.current_direction == Direction.DOWN:
            new_head_y += BLOCK_SIZE
        elif self.current_direction == Direction.UP:
            new_head_y -= BLOCK_SIZE

        # if the snake is out of the window, it appears at the other side
        # new_head_x %= WIN_WIDTH
        # new_head_y %= WIN_HEIGHT

        self.head = Block(new_head_x, new_head_y)
        self.snake_elements.insert(0, self.head)

    def draw(self, win):
        pygame.draw.rect(win, SNAKE_HEAD_COLOR, pygame.Rect(
            self.snake_elements[0].x, self.snake_elements[0].y, BLOCK_SIZE, BLOCK_SIZE))
        for block in self.snake_elements[1:]:
            pygame.draw.rect(win, SNAKE_COLOR, pygame.Rect(
                block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))

    def is_collided(self, block=None):
        if block == None:
            block = self.head

        if block in self.snake_elements[1:]:
            return True

        # hits boundary
        if block.x > WIN_WIDTH - BLOCK_SIZE or block.x < 0 or block.y > WIN_HEIGHT - BLOCK_SIZE or block.y < 0:
            return True

        return False

    def get_distance_from_apple(self, apple):
        return np.sqrt((self.head.x - apple.x)**2 + (self.head.y - apple.y)**2)


class Apple:

    def __init__(self):
        self.x, self.y = self.spawn_apple()

    def spawn_apple(self):  # TODO respawn if on snake
        x = random.randint(0, (WIN_WIDTH - BLOCK_SIZE) //
                           BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (WIN_HEIGHT - BLOCK_SIZE) //
                           BLOCK_SIZE) * BLOCK_SIZE

        return x, y

    def change_pos(self):
        self.x, self.y = self.spawn_apple()

    def draw(self, win):
        pygame.draw.rect(win, RED, pygame.Rect(
            self.x, self.y, BLOCK_SIZE, BLOCK_SIZE))


class SnakeGameAI:

    def __init__(self):
        self.width = WIN_WIDTH
        self.height = WIN_HEIGHT
        self.display = pygame.display.set_mode((self.width, self.height))
        self.frame = 0
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = Snake(self.width / 2, self.height / 2)
        self.apple = Apple()
        self.game_over = False
        self.score = 0
        self.frame = 0

    def play_step(self, action):
        self.frame += 1
        old_distance = self.snake.get_distance_from_apple(self.apple)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         action = np.array([0, 0, 1], dtype=np.int32)
            #     elif event.key == pygame.K_RIGHT:
            #         action = np.array([0, 1, 0], dtype=np.int32)

        self.snake.update(action)

        reward = 0
        if self.snake.is_collided():  # or self.frame > 100 * len(self.snake.snake_elements):
            self.game_over = True
            reward -= 100
            return self._get_state(), reward, self.game_over, self.score

        if self.snake.head.x == self.apple.x and self.snake.head.y == self.apple.y:
            self.score += 1
            reward += 10
            self.apple.change_pos()
        else:
            self.snake.snake_elements.pop()

        if self.snake.get_distance_from_apple(self.apple) < old_distance:
            reward += 5
        else:
            reward -= 5

        self.draw()
        self.clock.tick(GAME_SPEED)

        return self._get_state(), reward, self.game_over, self.score

    def draw(self):
        self.display.fill(BLACK)
        self.snake.draw(self.display)
        self.apple.draw(self.display)
        pygame.display.flip()

    def _get_state(self):
        snake_head = self.snake.snake_elements[0]

        block_left = Block(snake_head.x - BLOCK_SIZE, snake_head.y)
        block_right = Block(snake_head.x + BLOCK_SIZE, snake_head.y)
        block_up = Block(snake_head.x, snake_head.y - BLOCK_SIZE)
        block_down = Block(snake_head.x, snake_head.y + BLOCK_SIZE)

        is_direction_left = self.snake.current_direction == Direction.LEFT
        is_direction_right = self.snake.current_direction == Direction.RIGHT
        is_direction_up = self.snake.current_direction == Direction.UP
        is_direction_down = self.snake.current_direction == Direction.DOWN

        state = [
            # danger straight
            (is_direction_left and self.snake.is_collided(block_left)) or
            (is_direction_right and self.snake.is_collided(block_right)) or
            (is_direction_up and self.snake.is_collided(block_up)) or
            (is_direction_down and self.snake.is_collided(block_down)),

            # danger right
            (is_direction_left and self.snake.is_collided(block_up)) or
            (is_direction_right and self.snake.is_collided(block_down)) or
            (is_direction_up and self.snake.is_collided(block_right)) or
            (is_direction_down and self.snake.is_collided(block_left)),

            # danger left
            (is_direction_left and self.snake.is_collided(block_down)) or
            (is_direction_right and self.snake.is_collided(block_up)) or
            (is_direction_up and self.snake.is_collided(block_left)) or
            (is_direction_down and self.snake.is_collided(block_right)),

            # current direction
            is_direction_left,
            is_direction_right,
            is_direction_up,
            is_direction_down,

            # relative position to the apple
            self.apple.x < snake_head.x,  # apple left
            self.apple.x > snake_head.x,  # apple right
            self.apple.y < snake_head.y,  # apple up
            self.apple.y > snake_head.y  # apple down
        ]

        return np.array(state, dtype=int)
