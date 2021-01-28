`quadruped_motors.json` has all motors facing the same direction. E.g. all of the motors on the knee and hip flexion joints face in the body's positive x_hat direction.

`quadruped_motors_inverted.json` has the motors on the left side of the body facing in the body's negative x_hat direction, and the motors on the right side of the body facing in the body's positive x_hat direction.

`quadruped_motors.json` is designed to work with the quadruped callback. The `QuadrupedCallback` calculates joint angles as if all joints are facing the same direction.