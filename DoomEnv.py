import gym
from gym import spaces
import vizdoom as vzd
from time import sleep, time
import random
import numpy as np
import itertools as it
import copy
from PIL import Image as im
import skimage.color, skimage.transform
from debugging.keyboard import KEYBOARD

ENVS = ["basic", "deadly_corridor"]


class Rewards:
    def __init__(self, game):
        self.last_hits = 0
        self.last_kills = 0
        self.last_hitmon = 0

        varibales = game.get_available_game_variables()
        self.last_health = game.get_game_variable(varibales[0].HEALTH)

    def get_reward(self, current_hitmon, current_kill, current_hit, current_health, death, others):
        if death:
            death_val = 1
        else:
            death_val = 0
        info = {"hit": current_hit,
                "kill": current_kill,
                "hit monster": current_hitmon,
                "health": current_health,
                "death": death_val,
                "other": others}

        reward = (current_hit - self.last_hits) * (-2) + \
                 (current_kill - self.last_kills) * 100 + \
                 (current_hitmon - self.last_hitmon) * 1 + \
                 (current_health - self.last_health) * 20 + \
                 death_val * (-100) + \
                 others

        self.last_hits = current_hit
        self.last_health = current_health
        self.last_hitmon = current_hitmon
        self.last_kills = current_kill

        return reward, info


class DoomEnv(gym.Env):
    def __init__(self, display=False, feature="cnn", env_index: int = 0, debug=False, learning_type="sac"):
        self.config_dir = "./scenarios/" + ENVS[env_index] + ".cfg"

        self.display = display
        self.debug = debug
        self.game = self._initialize_game(config_file_path=self.config_dir)
        if self.debug:
            self.keyboard = KEYBOARD()

        n = self.game.get_available_buttons_size()
        # self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.actions = np.eye(n)

        # Sets time that will pause the engine after each action (in seconds)
        # Without this everything would go too fast for you to keep track of what's happening.
        if self.display:
            self.sleep_time = 1. / vzd.DEFAULT_TICRATE  # = 0.028
        else:
            self.sleep_time = .01 / vzd.DEFAULT_TICRATE  # = 0.0028

        self.resolution = (60, 90)
        self.feature = feature
        if self.feature == "cnn":
            self.observation_space = spaces.Box(low=0, high=255,  # dtype=np.uint8,
                                                shape=(self.resolution[0], self.resolution[1], 4))
        else:
            self.observation_space = spaces.Box(low=0, high=255,  # dtype=np.uint8,
                                                shape=(self.resolution[0] * self.resolution[1] * 4,))
        # self.action_space = spaces.Discrete(len(self.actions))

        self.learning_type = learning_type
        if self.learning_type == "sac":
            self.action_space = spaces.Box(np.array([0]), np.array([n]))
        elif self.learning_type == "ppo":
            self.action_space = spaces.Discrete(n)

        self.reward = Rewards(self.game)

        self.last_state = None
        self.last2n_state = None
        self.last3r_state = None

    def _initialize_game(self, config_file_path):
        game = vzd.DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(self.display)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        # game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.init()
        return game

    def reset(self):
        self.game.new_episode()
        self.last_state = None
        self.last2n_state = None
        self.last3r_state = None

        state, done, info = self._get_observation()

        self.reward = Rewards(self.game)
        return state

    def render(self, mode='human'):
        pass

    def preprocess(self, img):
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=2)
        return img

    def _get_observation(self):
        done = False
        if self.game.is_episode_finished():
            done = True

        # Gets the state
        state = self.game.get_state()
        obs = None
        if state is None:
            done = True
        else:
            # Which consists of:
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            # img = im.fromarray(automap_buf, 'RGB')
            # img.show("test")
            obs = self.preprocess(screen_buf)

            if self.last_state is None:
                self.last_state = copy.deepcopy(obs)
                self.last2n_state = copy.deepcopy(obs)
                self.last3r_state = copy.deepcopy(obs)

        if obs is None:
            observation = np.concatenate((self.last_state, self.last_state, self.last2n_state, self.last3r_state), axis=2)
            # observation = self.last_state
        else:
            # observation = obs
            observation = np.concatenate((obs, self.last_state, self.last2n_state, self.last3r_state), axis=2)
            self.last3r_state = copy.deepcopy(self.last2n_state)
            self.last2n_state = copy.deepcopy(self.last_state)
            self.last_state = copy.deepcopy(obs)

        if self.feature == "cnn":
            pass
        else:
            observation = np.asarray(observation).flatten()

        self.last_state = copy.deepcopy(obs)
        return observation, done, {"total_reward": self.game.get_total_reward()}

    def step(self, action, wait=False):
        if self.learning_type == "sac":
            action = int(action[0])
            if action == len(self.actions):
                action = len(self.actions) - 1
        r = self.game.make_action(self.actions[action].tolist())
        # r = self.game.make_action(self.actions[action])

        varibales = self.game.get_available_game_variables()
        reward, info_reward = self.reward.get_reward(current_hit=self.game.get_game_variable(varibales[0].HITS_TAKEN),
                                                     current_hitmon=self.game.get_game_variable(varibales[0].HITCOUNT),
                                                     current_kill=self.game.get_game_variable(varibales[0].KILLCOUNT),
                                                     current_health=self.game.get_game_variable(varibales[0].HEALTH),
                                                     death=self.game.get_game_variable(varibales[0].DEAD),
                                                     others=r)
        if self.sleep_time > 0 and wait:
            sleep(self.sleep_time)

        if self.debug:
            print("action: {}".format(action))
            print(info_reward)

        state, done, info = self._get_observation()

        return state, r, done, info

    def debug_run(self, total_time=1e10):
        start = time()
        while time() - start < total_time:
            key = self.keyboard.get_single_key(None)
            if key is not None and key < len(self.actions):
                state, reward, done, info = self.step(key, wait=True)

    def __del__(self):
        self.game.close()


if __name__ == "__main__":
    config_dir = "./scenarios/basic.cfg"
    # doom = DoomEnv(env_index=1, display=True, debug=True, learning_type="no")
    doom = DoomEnv(env_index=1, display=True, debug=False, learning_type="no")
    # doom.reset()
    # doom = SimpleDoomEnv(config_dir=config_dir)
    # done = False
    # while not done:
    #     doom.step(0)
    doom.debug_run(1e10)

# class SimpleDoomEnv(gym.Env):
#     def __init__(self,
#                  config_dir: str = "./scenarios/basic.wad",
#                  display: bool = False,
#                  total_time: int = 200):
#         self.game = vzd.DoomGame()
#         self._init_game(config_dir=config_dir, display=display, total_time=total_time)
#
#         # Define some actions. Each list entry corresponds to declared buttons:
#         # MOVE_LEFT, MOVE_RIGHT, ATTACK
#         # game.get_available_buttons_size() can be used to check the number of available buttons.
#         # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
#         self.actions = [[True, False, False], [False, True, False], [False, False, True]]
#
#         # Sets time that will pause the engine after each action (in seconds)
#         # Without this everything would go too fast for you to keep track of what's happening.
#         self.sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
#
#         self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
#         self.action_space = spaces.Discrete(len(self.actions))
#
#     def _init_game(self, config_dir, display, total_time):
#         self.game.set_doom_scenario_path(config_dir)
#
#         # Sets map to start (scenario .wad files can contain many maps).
#         self.game.set_doom_map("map01")
#
#         # Sets resolution. Default is 320X240, RES_640X480
#         self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
#
#         # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
#         self.game.set_screen_format(vzd.ScreenFormat.RGB24)
#
#         # Enables depth buffer.
#         self.game.set_depth_buffer_enabled(True)
#
#         # Enables labeling of in game objects labeling.
#         self.game.set_labels_buffer_enabled(True)
#
#         # Enables buffer with top down map of the current episode/level.
#         self.game.set_automap_buffer_enabled(True)
#
#         # Enables information about all objects present in the current episode/level.
#         self.game.set_objects_info_enabled(True)
#
#         # Enables information about all sectors (map layout).
#         self.game.set_sectors_info_enabled(True)
#
#         if display:
#             self._init_render()
#
#         # Adds buttons that will be allowed.
#         self.game.add_available_button(vzd.Button.MOVE_LEFT)
#         self.game.add_available_button(vzd.Button.MOVE_RIGHT)
#         self.game.add_available_button(vzd.Button.ATTACK)
#
#         # Adds game variables that will be included in state.
#         self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
#
#         # Causes episodes to finish after 200 tics (actions)
#         self.game.set_episode_timeout(total_time)
#
#         # Makes episodes start after 10 tics (~after raising the weapon)
#         self.game.set_episode_start_time(10)
#
#         # Sets the living reward (for each move) to -1
#         self.game.set_living_reward(-1)
#
#         # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
#         self.game.set_mode(vzd.Mode.PLAYER)
#
#         # Enables engine output to console.
#         # game.set_console_enabled(True)
#
#         # Initialize the game. Further configuration won't take any effect from now on.
#         self.game.init()
#
#     def reset(self):
#         self.game.new_episode()
#         state, done, info = self._get_observation()
#         return state
#
#     def _init_render(self):
#         # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
#         self.game.set_render_hud(False)
#         self.game.set_render_minimal_hud(False)  # If hud is enabled
#         self.game.set_render_crosshair(False)
#         self.game.set_render_weapon(True)
#         self.game.set_render_decals(False)  # Bullet holes and blood on the walls
#         self.game.set_render_particles(False)
#         self.game.set_render_effects_sprites(False)  # Smoke and blood
#         self.game.set_render_messages(False)  # In-game messages
#         self.game.set_render_corpses(False)
#         self.game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
#
#         # Makes the window appear (turned on by default)
#         self.game.set_window_visible(True)
#         # Turns on the sound. (turned off by default)
#         # game.set_sound_enabled(True)
#
#     def render(self, mode='human'):
#         pass
#
#     def _get_observation(self):
#         done = False
#         if self.game.is_episode_finished():
#             done = True
#
#         # Gets the state
#         state = self.game.get_state()
#         if state is None:
#             done = True
#         # Which consists of:
#         n = state.number
#         vars = state.game_variables
#         screen_buf = state.screen_buffer
#         depth_buf = state.depth_buffer
#         labels_buf = state.labels_buffer
#         automap_buf = state.automap_buffer
#         labels = state.labels
#         objects = state.objects
#         sectors = state.sectors
#
#         # img = im.fromarray(automap_buf, 'RGB')
#         # img.show("test")
#         return state, done, {"total_reward": self.game.get_total_reward()}
#
#     def step(self, action: int):
#         r = self.game.make_action(self.actions[action])
#
#         if self.sleep_time > 0:
#             sleep(self.sleep_time)
#
#         state, done, info = self._get_observation()
#         return state, r, done, info
#
#     def __del__(self):
#         self.game.close()
