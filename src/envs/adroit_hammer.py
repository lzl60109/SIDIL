import numpy as np
from d4rl.hand_manipulation_suite import HammerEnvV0

ADD_BONUS_REWARDS = True

class MyHammerEnvV0(HammerEnvV0):
    def __init__(self, random_start=False, terminate_on_success=False, **kwargs):
        self.random_start = random_start
        self.terminate_on_success = terminate_on_success
        super().__init__(**kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()
        
        # get to hammer
        reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
        # take hammer head to nail
        reward -= np.linalg.norm((tool_pos - target_pos))
        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if (np.linalg.norm(target_pos - goal_pos) < 0.020):
                reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < 0.010):
                reward += 75

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.010 else False

        if self.terminate_on_success:
            done = goal_achieved
        else:
            done = False

        return (
            ob, 
            reward, 
            False, 
            dict(goal_achieved=goal_achieved, is_success=goal_achieved)
        )


    def reset_model(self):
        self.sim.reset()

        target_bid = self.model.body_name2id('nail_board')
        self.model.body_pos[target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        
        self.sim.forward()
        return self.get_obs()
