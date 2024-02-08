# Plan A (Offline Framework) and Plan B (Online Framework)
## 7 terminals

## (1) Katla -- Pose detector
* setup virtual environment: `source ~/venvs/pose/bin/activate`
* source ros package:
```
cd ~/diver_id_ws2
source devel/setup.bash
```
* Run the detector: 
```
cd ~/diver_id_ws2/src/diver_joint/scripts/DEKR
./infer_ros.sh
```

## For tx2, we don't need venvs
## (2) tx2 -- DRP node
* (Required for Bench operation only): `rosservice call /mavros/cmd/arming "value: false"`
* Run the diver relative position node: 
```
cd adroc_diver_id_ws/
source devel/setup.bash
roslaunch adroc_diver drp_only.launch
```

## (3) tx2 -- Controller node
`roslaunch target_following reconfigure_drp_yaw_pitch_controller.launch` (don't need to source for this)

## (4) tx2 -- diverID node
```
cd adroc_diver_id_ws/
source devel/setup.bash
roslaunch adroc_diver aoc_only_planA.launch  (or, aoc_only_planB.launch)
```

## (5) tx2 -- initiate the action
```
cd adroc_diver_id_ws/
source devel/setup.bash
rostopic pub /adroc_diver/goal adroc_diver/ApproachDiverActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
  diver_id: 0" 
```
You can tab-complete this after `rostopic pub /adroc_diver/goal` (just make sure, the diverID algorithm is running)

## (6) tx2 -- record bag
```
cd adroc_diver_id_ws/
source devel/setup.bash
roslaunch adroc_diver data_record_test.launch my_args:="planX_caseY"
```

## (7) Katla -- rqt
```
rqt
/drp/drp_image (for PD view)
/detection/output_image (for pose detection)
```

# Plan C (Offline Data Collection + Offline Framework) 
## Offline Data Collection -- 4 terminals
## (1) Katla -- Pose detector
* setup virtual environment: `source ~/venvs/pose/bin/activate`
* source ros package:
```
cd ~/diver_id_ws2
source devel/setup.bash
```
* Run the detector: 
```
cd ~/diver_id_ws2/src/diver_joint/scripts/DEKR
./infer_ros.sh
```

## For tx2, we don't need venvs
## (2) tx2 -- DRP node
* (Required for Bench operation only): `rosservice call /mavros/cmd/arming "value: false"`
* Run the diver relative position node: 
```
cd adroc_diver_id_ws/
source devel/setup.bash
roslaunch adroc_diver drp_only.launch
```

## (3) tx2 -- feature collection
* Before running this script, change `DIVER_ID` value to reflect the current diver's assigned class. Lowest value is `0` and maximum value is `DIVERS_NUM-1`. Make sure to change this everytime, otherwise the collected data and labels will be overwritten.
```
cd adroc_diver_id_ws/
source devel/setup.bash
rosrun adroc_diver individual_feature_collect_diverID.py
```

## (4) tx2 -- record bag
```
cd adroc_diver_id_ws/
source devel/setup.bash
roslaunch adroc_diver data_record_test.launch my_args:="planC_collect"
```

## Offline Training
* change the `DIVERS_NUM` variable to reflect the number of divers
* do this for different models and dataset by changing `self.model` and `self.data_type` variable
```
cd adroc_diver_id_ws/src/adroc_diver/scripts
python3 OfflineModelTraining.py
```

## Inference Step
* change the `self.TARGET_DIVER` (inside the __init__ function) variable to set specific target.
* perform everything as Plan A/B, but run planC launch file instead
```
cd adroc_diver_id_ws/
source devel/setup.bash
roslaunch adroc_diver aoc_only_planC.launch
```

# Troubleshooting/Parameter Tuning
## when robot pitch is flipped
```
vim /home/irvlab/catkin_ws/src/target_following/scripts/drp_yaw_pitch_controller_reconf.py
```
    def set_vyprh_cmd(self, ss, yy, pp, rr, hh):
        self.cmd_msg.throttle = ss
        self.cmd_msg.yaw = yy
        self.cmd_msg.pitch = pp # <---- change this to -pp
        #self.cmd_msg.roll = rr
        #self.cmd_msg.heave = hh

* you can slow down the robot by dividing `ss, yy, pp` with some numbers

## Set up image set point for the robot to aim towards
To achieve this, you need to modify TWO scripts
* modify the controller node:
```
vim /home/irvlab/catkin_ws/src/target_following/scripts/drp_yaw_pitch_controller_reconf.py
```
    def compute_errors_from_estimate(self): 
        tx, ty, pd = self.current_observation

        #Our image target point is centered horizontally, and 1/3 of the way down vertically.
        image_setpoint_x = self.image_w/2.0 
        image_setpoint_y = self.image_h/2.0 # <---- change this, ie, divide with a larger number to move the image setpoint up

* modify the diverID node
```
vim /home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/autonomous_operator_configuration_deploy_planA.py (or, planB/planC)
```
    def drp_stable(self):
        if len(self.drp_msgs) >0:
            x_errs = list()
            y_errs = list()
            pd_errs = list()

            image_setpoint_x = self.drp_msgs[0].image_w/2.0
            image_setpoint_y = self.drp_msgs[0].image_h/2.0 # <---- change this, ie, divide with the SAME number

## If you want to change the DRP target for the detected pose
```
vim /home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver_relative_postion_deploy.py
```
    def pose_to_drp(self):
        cp_x, cp_y, pd = None, None, None

        rx, ry = self.rs_observation
        lx, ly = self.ls_observation
        rhx, rhy = self.rh_observation
        lhx, lhy = self.lh_observation

        cp_x = int((lx+rx+lhx+rhx)/4) # <---- change this to define a new target location on the body
        cp_y = int((ly+ry+lhy+rhy)/4) # <---- change this to define a new target location on the body
        
## Pseudo-distance (`pd`) Definition and How to change the `pd` value?
* If `pd > 1` --> robot will back up; if `pd < 1` --> robot will come close.
* If `shoulder_to_target_ratio` is increased, then `pd` would be low, so robot comes closer.
* If `shoulder_to_target_ratio` is decreased, then `pd` would be high, so robot goes backward.
* To change the `shoulder_to_target_ratio`, open
```
vim /home/irvlab/adroc_diver_id_ws/src/adroc_diver/scripts/diver_relative_postion_deploy.py
```
        self.bbox_target_ratio = 0.17*1.5 # <---- change this value to incorporate T-pose bbox shape (need to larger than this)
        self.shoulder_target_ratio= 0.17 # <---- change this value

## ROS log removal
    rosclean check
    rosclean purge
