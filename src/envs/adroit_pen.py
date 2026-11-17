import numpy as np
from d4rl.hand_manipulation_suite import PenEnvV0
from d4rl.utils.quatmath import quat2euler, euler2quat

ADD_BONUS_REWARDS = True

class MyPenEnvV0(PenEnvV0):
    def __init__(self, random_start=False, terminate_on_success=False, **kwargs):
        self.random_start = random_start
        self.terminate_on_success = terminate_on_success
        super().__init__(**kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            starting_up = False
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            starting_up = True
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos-desired_loc)
        reward = -dist
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)
        reward += orien_similarity

        if ADD_BONUS_REWARDS:
            # bonus for being close to desired orientation
            if dist < 0.075 and orien_similarity > 0.9:
                reward += 10
            if dist < 0.075 and orien_similarity > 0.95:
                reward += 50

        # penalty for dropping the pen
        done = False
        if obj_pos[2] < 0.075:
            reward -= 5
            done = True if not starting_up else False

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        if self.terminate_on_success:
            done = goal_achieved
        else:
            done = False

        return (
            self.get_obs(), 
            reward, 
            done, 
            dict(goal_achieved=goal_achieved, is_success=goal_achieved)
        )


    def reset_model(self):
        qp = self.init_qpos.copy()

        # qp
        if self.random_start:
            # Randomize all qpos
            qp += self.np_random.uniform(low=-0.1, high=0.1, size=qp.shape)

        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        
        self.sim.forward()
        return self.get_obs()
