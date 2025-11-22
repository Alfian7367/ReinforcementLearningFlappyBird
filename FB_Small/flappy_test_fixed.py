import pygame, random, time, sys
from pygame.locals import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# --- VARIABLES ---
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 10 
GRAVITY = 1
GAME_SPEED = 5 # Must match speed used in training

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT= 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

# --- FIXED PATTERN CONFIGURATION ---
# Pattern: Middle (250) -> Top (350) -> Middle (250) -> Bottom (150) -> Repeat
PIPE_PATTERN = [250, 350, 250, 150] 
pipe_cycle_index = 0

pygame.mixer.init()

# --- GAME CLASSES ---

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        # Ensure display is initialized for asset loading (for safety)
        if not pygame.display.get_init():
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            
        self.images = [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.reset()

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def reset(self):
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2
        self.speed = 0

class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)
        self.passed = False

    def update(self):
        self.rect[0] -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED

def get_fixed_pipes(xpos):
    """Generates pipes based on the deterministic PIPE_PATTERN."""
    global pipe_cycle_index
    
    # Get current size from pattern
    size = PIPE_PATTERN[pipe_cycle_index]
    
    # Increment and wrap around
    pipe_cycle_index = (pipe_cycle_index + 1) % len(PIPE_PATTERN)
    
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

class GameEnvironment:
    def __init__(self, render=True):
        pygame.init()
        self.render = render

        # CRITICAL FIX: Initialize display context for asset loading
        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird AI - FIXED PATTERN TEST')
        else:
            self.screen = pygame.display.set_mode((1, 1), pygame.HIDDEN)
        
        # Load assets based on render flag
        if self.render:
            self.BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
            self.BACKGROUND = pygame.transform.scale(self.BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
            self.score_font = pygame.font.SysFont('Consolas', 40, bold=True)
            self.clock = pygame.time.Clock()

        self.bird_group = pygame.sprite.Group()
        self.bird = Bird() 
        self.bird_group.add(self.bird)

        self.ground_group = pygame.sprite.Group()
        for i in range (2):
            ground = Ground(GROUND_WIDHT * i)
            self.ground_group.add(ground)

        self.pipe_group = pygame.sprite.Group()
        
        self.score = 0
        self.reset() 

    def reset(self):
        """Resets the game state and pipe pattern."""
        self.bird.reset()
        self.score = 0
        
        # Reset pattern index so every test run is identical
        global pipe_cycle_index
        pipe_cycle_index = 0
        
        self.pipe_group.empty()
        for i in range (2):
            # Use the fixed pipe generator
            pipes = get_fixed_pipes(SCREEN_WIDHT + (i * SCREEN_WIDHT / 1.5)) 
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
            
        return self._get_state()

    def _get_next_pipe(self):
        """Helper to find the next upcoming pipe."""
        min_x = float('inf')
        next_pipe_bottom = None
        next_pipe_top = None

        bottom_pipes = [p for p in self.pipe_group if p.rect.bottom > SCREEN_HEIGHT / 2]
        
        for pipe in sorted(bottom_pipes, key=lambda p: p.rect[0]):
            if pipe.rect[0] > self.bird.rect[0] - PIPE_WIDHT:
                min_x = pipe.rect[0]
                # Find its corresponding top pipe
                for p_top in self.pipe_group:
                    if p_top.rect[0] == min_x and p_top.rect.bottom < SCREEN_HEIGHT / 2:
                        next_pipe_top = p_top
                        break
                next_pipe_bottom = pipe
                break
        
        if next_pipe_bottom is None:
            return None, None

        return next_pipe_bottom, next_pipe_top

    def _get_state(self):
        """Returns 5 normalized state features (MUST match training model input)."""
        next_pipe_bottom, next_pipe_top = self._get_next_pipe()
        
        # Fallback state if no pipe is visible
        if next_pipe_bottom is None or next_pipe_top is None:
            return np.array([
                self.bird.rect[1] / SCREEN_HEIGHT,
                self.bird.speed / 100,      # Normalized Bird Vertical Velocity
                1.0,                      # Horizontal Distance (Far away)
                0.5,                        # Vertical Distance (Middle of screen)
                0.5                         # Gap Y-position (Middle of screen)
            ])

        horizontal_dist = (next_pipe_bottom.rect[0] - self.bird.rect[0]) / SCREEN_WIDHT
        gap_center_y = (next_pipe_bottom.rect.top + next_pipe_top.rect.bottom) / 2
        vertical_dist = (gap_center_y - self.bird.rect[1]) / SCREEN_HEIGHT
        
        state = np.array([
            self.bird.rect[1] / SCREEN_HEIGHT,        # 1. Bird Y Position (Normalized)
            self.bird.speed / 100,                    # 2. Bird Vertical Velocity (Normalized)
            horizontal_dist,                          # 3. Horizontal distance to pipe (Normalized)
            vertical_dist,                            # 4. Vertical distance to gap center (Normalized)
            gap_center_y / SCREEN_HEIGHT              # 5. Absolute Normalized Gap Center Y Position (Normalized)
        ])
        
        return state

    def step(self, action):
        if action == 1:
            self.bird.bump()

        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()

        done = False
        reward = 0 
        
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask)):
            done = True
            return self._get_state(), -1, done # Return a simple crash signal
        
        if self.bird.rect.top < 0:
            done = True
            return self._get_state(), -1, done # Crash signal

        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        if is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            
            # Use the fixed pipe generator here
            pipes = get_fixed_pipes(SCREEN_WIDHT * 2 - 20)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
        
        for pipe in self.pipe_group:
            if pipe.rect.right < self.bird.rect.left and not pipe.passed:
                pipe.passed = True
                if pipe.rect.bottom > SCREEN_HEIGHT / 2: 
                    self.score += 1
                    reward = 1
        
        if self.render:
            self._draw_screen()
            # Set tick higher to make AI run faster during viewing
            self.clock.tick(60) 
        
        return self._get_state(), reward, done

    def _draw_screen(self):
        self.screen.blit(self.BACKGROUND, (0, 0))
        self.bird_group.draw(self.screen)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)
        
        score_text = self.score_font.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        pattern_text = self.score_font.render("FIXED PATTERN TEST", True, (255, 100, 100))
        
        self.screen.blit(pattern_text, (10, 10))
        self.screen.blit(score_text, (10, 50))
        
        pygame.display.update()
        
# --- AGENT FOR INFERENCE (EXPLOITATION ONLY) ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        """Model must match the 5-input, 64-neuron architecture used for training."""
        model = keras.Sequential()
        # Ensure input_dim is 5 and neurons are 64
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def act(self, state):
        # Pure exploitation: always pick the action with the highest predicted Q-value
        act_values = self.model(state, training=False)
        return np.argmax(act_values.numpy()[0])

    def load(self, name):
        if os.path.exists(name):
            self.model.load_weights(name)
            print(f"SUCCESS: Loaded weights from {name}. Ready to test.")
        else:
            print(f"ERROR: Could not find weight file: {name}. Please train the model first.")
            sys.exit()

if __name__ == "__main__":
    
    env = GameEnvironment(render=True) # Render = True to watch the AI fly
    
    state_size = 5 # Must match training input size
    action_size = 2
    
    agent = DQNAgent(state_size, action_size)
    
    # Load the trained model. This requires the flappy-dqn.weights.h5 file to exist!
    agent.load("flappy-dqn.weights.h5")
    
    print("\n--- STARTING FIXED PATTERN TEST RUN ---")
    print(f"Pattern: {PIPE_PATTERN}. Press ESC to quit.")
    
    episodes = 10 # Run 10 test games
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            if done:
                print(f"Test Game {e+1}: Score: {env.score}")