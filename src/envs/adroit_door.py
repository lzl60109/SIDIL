import numpy as np
from d4rl.hand_manipulation_suite import DoorEnvV0

ADD_BONUS_REWARDS = True


class MyDoorEnvV0(DoorEnvV0):
    def __init__(self, random_start=False, terminate_on_success=False, **kwargs):
        self.random_start = random_start
        self.terminate_on_success = terminate_on_success
        super().__init__(**kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False

        if self.terminate_on_success:
            done = goal_achieved
        else:
            done = False

        return (
            ob,
            reward,
            done,
            dict(goal_achieved=goal_achieved, is_success=goal_achieved),
        )

    def reset_model(self):
        qp = self.init_qpos.copy()

        # qp
        if self.random_start:
            # Randomize all qpos
            qp += self.np_random.uniform(low=-0.1, high=0.1, size=qp.shape)

        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
 

        self.sim.forward()
        return self.get_obs()
