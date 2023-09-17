import math
import numpy as np
import ipdb as pdb
from scipy.interpolate import interpn

def toLocal(x, y, v, yaw, origin, local_race_yaw):
    ## TODO: add v and yaw
    '''
    # v' = v
    # yaw' = yaw - yaw0
    '''
    yaw = yaw-local_race_yaw
    # (x, y) -> (x', y')
    XY = np.array([x,y])
    transform = np.array([[np.cos(local_race_yaw), np.sin(local_race_yaw)], 
                  [-np.sin(local_race_yaw), np.cos(local_race_yaw)]])
    coords = (XY-origin).dot(transform.T)
    
    return np.array(np.concatenate([coords, [v, yaw]]))

def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
    return yaw

class SafeController():
    def __init__(self, 
            x=[0, 0, 0, 0],
            uMin=[-1.0, -1.0], # [\delta, a]
            uMax=[1.0, 1.0],
            dMin=[0.0, 0.0],
            dMax=[0.0, 0.0],
            uMode="max", 
            verbose = False, 
            margin = 1):
        self.x = x
        self.uMax = uMax 
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        assert uMode in ["min", "max"]
        self.uMode = uMode
        if uMode == "min":
            dMode = "max"
        else:
            dMode = "min"
        self.dMode = dMode
        self.verbose = verbose
        self.margin = margin

    def select_action(self, local_state, safety):
        """
        Input: 
            local_state:
            safety_set: 
        Output: 
            action: [steering, acc]
        """
        ## esimate dV/dx with finite difference 
        x, y, v, yaw = local_state.tolist()
        grid = [safety['x'], safety['y'], safety['v'], safety['yaw']]
        
        # V_current = interpn(grid,  safety['V'], local_state)
        J_v = safety['J3']
        J_yaw = safety['J4']
        #V_current = interpn(grid, safety['V'], local_state)   
        
        ## Already clipped before passing back
        '''
        min_state = np.array([min(safety['x']), min(safety['y']), min(safety['v']), min(safety['yaw'])])
        max_state = np.array([max(safety['x']), max(safety['y']), max(safety['v']), max(safety['yaw'])])
        local_state = np.clip(local_state, a_min=min_state, a_max=max_state)
        '''
        try:   
            J_yaw = interpn(grid,  J_yaw, local_state) 
        except:
            J_yaw = 0
        try:    
            J_v = interpn(grid,  J_v, local_state) 
        except:
            J_v = -1

        if J_yaw < 0:
            opt_w = self.uMin[0]
            w_msg = 'RIGHT'
        elif J_yaw == 0:
            opt_w = 0
            w_msg = '-'
        else:
            opt_w = self.uMax[0]
            w_msg = 'LEFT'
        if J_v < 0:
            opt_a = self.uMin[1] 
            a_msg = 'BRAKE'
        elif J_v == 0:
            opt_a = 0
            a_msg = '-'
        else:
            opt_a = self.uMax[1]
            a_msg = 'ACC'
        '''
        ## Prevent the braking from being too conservative
        if ((v<3) & (V_current>3)) | ((v<1) & (V_current>1.8)):
            opt_a = 0
        '''
        ## Prevent the vehicle from coming to a complete stop
        if v<0.3:
            opt_a = 0.1
        
        ## Prevent the steering angle from being too large
        # v in m/s; 1/s->2.24mph
        if v > 30:
            opt_w = np.clip(opt_w, a_min=-1/12, a_max=1/12)
        elif v > 20:
            opt_w = np.clip(opt_w, a_min=-1/6, a_max=1/6)
        elif v > 10:
            opt_w = np.clip(opt_w, a_min=-1/3, a_max=1/3)
        
        if self.verbose:
            #print(f"Safe Controller: {'LEFT' if opt_w>0 else 'RIGHT'}")
            print(f"Safe Controller: {w_msg}, {a_msg}")
        return np.array([opt_w, opt_a])
