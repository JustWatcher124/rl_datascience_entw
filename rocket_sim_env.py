import gymnasium as gym
from gymnasium import spaces
import numpy as np
import RocketSim as rs

from env_utils import get_car_state, get_ball_state, calculate_flying_reward, check_done
import csv
import os

class RocketSimFlyingEnv(gym.Env):
    def __init__(self, verbosity=0, log_csv_file="/tmp/gym/rocket_sim_flying_env_log.csv"):
        super(RocketSimFlyingEnv, self).__init__()
        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.car = self.arena.add_car(rs.Team.BLUE)
        self.reset_car()
        self.verbosity = verbosity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.movement_ticks = 0
        self.step_counter = 0
        self.total_steps = 0
        self.log_csv_file = log_csv_file

    def seed(self, seed=None):
        return [69]
    
    def reset_car(self):
        self.car.set_state(rs.CarState(pos=rs.Vec(z=17.03), vel=rs.Vec(), boost=100.0))
        self.arena.ball.set_state(rs.BallState(pos=rs.Vec(z=1000), vel=rs.Vec()))
        self.movement_ticks = 0
        self.step_counter = 0

   
    def reset(self, *, seed=None):
        super().reset(seed=seed)

        self.reset_car()
        obs = self._get_obs()
        return obs, {}
       
       

    def _get_obs(self):
        car_state = get_car_state(self.car)
        ball_state = get_ball_state(self.car)
        model_input = np.concatenate((car_state, ball_state), axis=1)
        return model_input.flatten().astype(np.float32)

    def step(self, action):
        
        car_state = self.car.get_state()  # give the car 100% of boost
        car_state.boost = 100.0
        self.car.set_state(car_state)


        throttle, steer, pitch, yaw, roll = action[:5]
        boost, jump, handbrake = action[5:] > 0  # thresholding binary actions

        controls = rs.CarControls(
            throttle=throttle, steer=steer, pitch=pitch, yaw=yaw, roll=roll,
            boost=bool(boost), jump=bool(jump), handbrake=bool(handbrake)
        )
        self.car.set_controls(controls)

        # Simulate one tick
        self.arena.step(1)

        state = self._get_obs()
        reward = calculate_flying_reward(get_car_state(self.car), time_held=self.step_counter)
        # print(reward)
        done, self.movement_ticks = check_done(get_car_state(self.car), self.movement_ticks)
        if self.step_counter < 500:
            done = False
        self.step_counter += 1
        self.total_steps += 1
        wheels_with_contact = state[-4:]
        is_flying = np.sum(wheels_with_contact) == 0
        
        if self.verbosity > 0:
            x, y, z = state[0], state[1], state[2]
            vx, vy, vz = state[3], state[4], state[5]
            velocity = np.linalg.norm([vx, vy, vz])
            print(f"Step {self.step_counter} | Total Steps: {self.total_steps} | "
              f"Position: ({x:.2f}, {y:.2f}, {z:.2f}) | "
              f"Velocity: {velocity:.2f} | "
              f"Reward: {reward:.2f} | "
              f"Done: {done} | "
              f"Is Flying: {is_flying}", end="\r")
            log_fields = [
            "step", "total_steps", "x", "y", "z", "vx", "vy", "vz",
            "velocity", "reward", "done", "is_flying"
            ]
            log_row = [
            self.step_counter, self.total_steps, x, y, z, vx, vy, vz,
            velocity, reward, done, is_flying
            ]
            file_exists = os.path.isfile(self.log_csv_file)
            with open(self.log_csv_file, mode="a", newline="") as csvfile:
        
                writer = csv.writer(csvfile)
                if not file_exists or os.path.getsize(self.log_csv_file) == 0:
                    writer.writerow(log_fields)
                writer.writerow(log_row)
        return state, reward, done, False, {}

