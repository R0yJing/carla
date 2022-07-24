import math
AUGMENT_LATERAL_STEERINGS = 6
def augment_steering(camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras.
        Args:
            camera_angle_batch: in degrees
            steer_batch:
            speed_batch:
        Returns:
            the augmented steering
        """

        time_use = 1.0
        car_length = 6.0
        old_steer = steer
        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = AUGMENT_LATERAL_STEERINGS * (
        math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))


        #print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
        return steer
small_spd = []
normal_spd = []
for i in range(30):
    normal_spd.append(augment_steering(-35,-i/30,10))
    small_spd.append(augment_steering(-35, -i/30,10/30))
   
