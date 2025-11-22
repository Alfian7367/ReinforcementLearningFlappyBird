import pygame, random, time, sys
from pygame.locals import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque
import json
import csv  # Added for CSV logging
import os   # Added for file checking

#VARIABLES
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 10 
GRAVITY = 1
GAME_SPEED = 5

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT= 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

# Pattern: Middle -> Top -> Middle -> Bottom -> Repeat
PIPE_PATTERN = [250, 350, 250, 150] 
pipe_cycle_index = 0

pygame.mixer.init()

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
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

def get_random_pipes(xpos):
    # --- MODIFICATION: Random Height Restored ---
    size = random.randint(100, 300)
    
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

class GameEnvironment:
    def __init__(self, render=True):
        pygame.init()
        self.render = render

        self.screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
        
        if self.render:
            pygame.display.set_caption('Flappy Bird AI - Random Small Optimized')
            self.BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
            self.BACKGROUND = pygame.transform.scale(self.BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
            self.score_font = pygame.font.SysFont('Consolas', 40, bold=True)
            self.clock = pygame.time.Clock()

        # Create sprite groups
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
        """Resets the game state for a new episode."""
        self.bird.reset()
        self.score = 0
        
        # Reset pipes
        self.pipe_group.empty()
        for i in range (2):
            pipes = get_random_pipes(SCREEN_WIDHT + (i * SCREEN_WIDHT / 1.5)) # Spaced out pipes
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
                # Now find its corresponding top pipe
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
        next_pipe_bottom, next_pipe_top = self._get_next_pipe()
        
        if next_pipe_bottom is None or next_pipe_top is None:
            return np.array([
                self.bird.rect[1] / SCREEN_HEIGHT,
                self.bird.speed / SPEED,
                1.0,  
                0.5   
            ])

        horizontal_dist = (next_pipe_bottom.rect[0] - self.bird.rect[0]) / SCREEN_WIDHT
        gap_center_y = (next_pipe_bottom.rect.top + next_pipe_top.rect.bottom) / 2
        vertical_dist = (gap_center_y - self.bird.rect[1]) / SCREEN_HEIGHT
        
        state = np.array([
            self.bird.rect[1] / SCREEN_HEIGHT,
            self.bird.speed / (SPEED * 2), 
            horizontal_dist,
            vertical_dist
        ])
        
        return state

    def step(self, action):
        if action == 1:
            self.bird.bump()

        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()

        done = False
        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask)):
            done = True
            reward = -1.0 # Normalized penalty for dying
            return self._get_state(), reward, done
        
        if self.bird.rect.top < 0:
            done = True
            reward = -0.5 # Normalized penalty for flying too high
            return self._get_state(), reward, done

        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        reward = 0.005 # Small normalized reward for surviving
        
        if is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            
            # Random pipes generated here
            pipes = get_random_pipes(SCREEN_WIDHT * 2 - 20)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
        
        for pipe in self.pipe_group:
            if pipe.rect.right < self.bird.rect.left and not pipe.passed:
                pipe.passed = True
                if pipe.rect.bottom > SCREEN_HEIGHT / 2: 
                    self.score += 1
                    reward = 1.0 # Normalized reward for passing a pipe
        
        if self.render:
            self._draw_screen()
            self.clock.tick(30) 
        
        return self._get_state(), reward, done

    def _draw_screen(self):
        self.screen.blit(self.BACKGROUND, (0, 0))
        self.bird_group.draw(self.screen)
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)
        
        score_text = self.score_font.render(str(int(self.score)), True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(SCREEN_WIDHT / 2, 50))
        self.screen.blit(score_text, score_rect)
        
        pygame.display.update()
        
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() 

    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model(state, training=False)
        return np.argmax(act_values.numpy()[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 

        minibatch = random.sample(self.memory, batch_size)
        
        states = np.squeeze(np.array([t[0] for t in minibatch]))
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.squeeze(np.array([t[3] for t in minibatch]))
        dones = np.array([t[4] for t in minibatch])

        q_values_current = self.model(states, training=False).numpy()
        q_values_next = self.target_model(next_states, training=False).numpy()

        targets = q_values_current.copy()

        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(q_values_next[i])

        self.model.train_on_batch(states, targets)
            
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model() 

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    
    RENDER_GAME = True 
    
    env = GameEnvironment(render=RENDER_GAME)
    
    state_size = 4
    action_size = 2
    
    agent = DQNAgent(state_size, action_size)
    
    EPISODES = 10000
    BATCH_SIZE = 10
    
    global_frame_counter = 0
    TRAIN_EVERY_N_FRAMES = 30 
    UPDATE_TARGET_EVERY_N_EPISODES = 5 
    
    # --- NEW: CSV Logging Setup ---
    CSV_FILENAME = "training_log.csv"
    
    # Write header only if file doesn't exist
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Score", "Epsilon", "Total Reward"])
    
    try:
        agent.load("flappy-dqn.weights.h5")
        agent.epsilon = agent.epsilon_min
        print("Loaded model weights.")
    except:
        print("No saved model found, starting new training.")

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        done = False
        while not done:
            global_frame_counter += 1
            
            if RENDER_GAME:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode: {e+1}/{EPISODES}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}, Total Reward: {total_reward:.2f}")
                
                # --- NEW: Write to CSV ---
                with open(CSV_FILENAME, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([e+1, env.score, f"{agent.epsilon:.4f}", f"{total_reward:.2f}"])
                break

            if global_frame_counter % TRAIN_EVERY_N_FRAMES == 0:
                agent.replay(BATCH_SIZE)
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % UPDATE_TARGET_EVERY_N_EPISODES == 0:
            agent.update_target_model()
            print(f"--- Updated Target Model (Episode {e}) ---")

        if e % 10 == 0 and e > 0:
            agent.save("flappy-dqn.weights.h5")
            print(f"Saved model weights at episode {e}")
            
            metadata = {
                'total_episodes_run': e,
                'current_epsilon': agent.epsilon
            }

            with open("flappy-dqn.json", "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved training metadata (episode {e}) to flappy-dqn.json")