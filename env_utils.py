import numpy as np


def get_car_state(car):
    """
    Extract the car's state from a RocketSim car object as a numpy array.

    The returned array contains the following features:
        - Position (x, y, z)
        - Velocity (x, y, z)
        - Rotation matrix (first row: 3 values)
        - Angular velocity (x, y, z)
        - Last relative dodge torque (x, y, z)
        - Current flip relative torque (x, y, z)
        - Last controls: throttle, steer, pitch, yaw, roll, boost, jump, handbrake
        - World contact normal (x, y, z)
        - Wheels with contact (4 values, as integers)

    Args:
        car: RocketSim car object.

    Returns:
        np.ndarray: Array of car state features with shape (1, N).
    """
    car_pos = car.get_state().pos
    car_vel = car.get_state().vel
    car_rot_mat = car.get_state().rot_mat
    car_ang_vel = car.get_state().ang_vel
    car_last_rel_dodge_torque = car.get_state().flip_rel_torque
    car_flip_rel_torque = car.get_state().flip_rel_torque
    car_last_controls = car.get_state().last_controls
    car_world_contact_normal = car.get_state().world_contact_normal
    car_wheels_with_contact = car.get_state().wheels_with_contact

    car_state = np.array([
        car_pos.x, car_pos.y, car_pos.z,
        car_vel.x, car_vel.y, car_vel.z,
        car_rot_mat[0][0], car_rot_mat[0][1], car_rot_mat[0][2],
        car_ang_vel.x, car_ang_vel.y, car_ang_vel.z,
        car_last_rel_dodge_torque.x, car_last_rel_dodge_torque.y, car_last_rel_dodge_torque.z,
        car_flip_rel_torque.x, car_flip_rel_torque.y, car_flip_rel_torque.z,
        car_last_controls.throttle, car_last_controls.steer, 
        car_last_controls.pitch, car_last_controls.yaw, 
        car_last_controls.roll, int(car_last_controls.boost), 
        int(car_last_controls.jump), int(car_last_controls.handbrake),
        int(car_world_contact_normal.x), int(car_world_contact_normal.y), int(car_world_contact_normal.z),
        *map(lambda x: int(x), car_wheels_with_contact)
    ]).reshape(1, -1)
    return car_state


def get_ball_state(car):
    """
    Extracts ball hit information from the RocketSim car object and returns it as a numpy array.

    The returned array contains:
        - Relative position on ball (x, y, z)
        - Ball position (x, y, z)
        - Extra hit velocity (x, y, z)
        - Tick count when hit
        - Tick count when extra impulse applied

    Args:
        car: RocketSim car object.

    Returns:
        np.ndarray: Array of ball hit features with shape (1, 11).
    """
    ball_hit_info = car.get_state().ball_hit_info
    rel_pos_on_ball = ball_hit_info.relative_pos_on_ball  # Vec: .x, .y, .z
    ball_pos = ball_hit_info.ball_pos             # Vec: .x, .y, .z
    extra_vel = ball_hit_info.extra_hit_vel       # Vec: .x, .y, .z
    tick_hit = ball_hit_info.tick_count_when_hit  # int
    tick_impulse = ball_hit_info.tick_count_when_extra_impulse_applied  # int
    rel_pos_on_ball = ball_hit_info.relative_pos_on_ball  # Vec: .x, .y, .z
    extra_vel = ball_hit_info.extra_hit_vel             # Vec: .x, .y, .z
    tick_hit = ball_hit_info.tick_count_when_hit        # int
    tick_impulse = ball_hit_info.tick_count_when_extra_impulse_applied  # int

    ball_state = np.array([
        rel_pos_on_ball.x, rel_pos_on_ball.y, rel_pos_on_ball.z,
        ball_pos.x, ball_pos.y, ball_pos.z,
        extra_vel.x, extra_vel.y, extra_vel.z,
        tick_hit, tick_impulse
    ]).reshape(1, -1)
    return ball_state


def calculate_flying_reward(car_state, velocity_threshold=100, max_reward=1000, time_held=1):
    """
    Calculates a reward based on car velocity, flying state, and time held.
    - car_state: output from get_car_state (shape: (1, 33))
    - velocity_threshold: velocity (in units) below which reward is maximized
    - max_reward: maximum possible reward
    - time_held: number of consecutive steps the condition is held
    Returns: float (reward)
    """
    # Extract velocity components
    vx, vy, vz = car_state[0, 3], car_state[0, 4], car_state[0, 5]
    velocity = np.linalg.norm([vx, vy, vz])
    
    # Wheels with contact: last 4 elements of car_state
    wheels_with_contact = car_state[0, -4:]
    is_flying = np.sum(wheels_with_contact) == 0
    if is_flying:
        # Exponential reward: lower velocity = higher reward
        reward = max_reward * np.exp(-velocity / velocity_threshold)
        # Scale by time held (e.g., linearly or with sqrt)
        reward *= np.sqrt(time_held)
        return reward
    else:
        return 0.0

def check_done(car_state, movement_ticks, velocity_threshold=100, max_allowed_velocity=600, required_ticks=1200):
    """
    Determines if the game is done based on car movement.
    - car_state: output from get_car_state (shape: (1, 33))
    - movement_ticks: current count of ticks with velocity > threshold
    - velocity_threshold: minimum velocity to count as movement
    - required_ticks: number of ticks (at 120/s) to trigger done
    Returns: (done: bool, movement_ticks: int)
    """
    vx, vy, vz = car_state[0, 3], car_state[0, 4], car_state[0, 5]
    velocity = np.linalg.norm([vx, vy, vz])
    if velocity > max_allowed_velocity:
        return True, movement_ticks  # Game over if velocity exceeds max allowed
    
    # Increment movement ticks if velocity is above threshold
    if velocity > velocity_threshold:
        movement_ticks += 1
    else:
        movement_ticks = 0  # Reset if below threshold
    # Check if velocity was too high for too long
    done = movement_ticks >= required_ticks
    return done, movement_ticks