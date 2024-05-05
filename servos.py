from gpiozero import AngularServo
from time import sleep

# Define the GPIO pins for the servos
phi_pin = 16
theta_pin = 21

# Create servo objects
phi_servo = AngularServo(phi_pin, min_angle=-90, max_angle=90)
theta_servo = AngularServo(theta_pin, min_angle=-90, max_angle=90)

# List of [phi, theta, timeout] coordinates
movements = [
    [0, 0, 1],
    [30, 30, 2],
    [-30, -30, 2],
    [45, -45, 1],
    [-45, 45, 1],
    [0, 0, 1]
]

def smooth_move_servo(servo, start_angle, end_angle, duration):
    # Calculate the number of steps
    steps = 50  # More steps for smoother movement
    step_duration = duration / steps
    step_angle = (end_angle - start_angle) / steps
    
    for step in range(steps):
        servo.angle = start_angle + step * step_angle
        sleep(step_duration)
    servo.angle = end_angle  # Ensure it ends at the exact target angle

def move_servos(movements):
    for phi_angle, theta_angle, pause in movements:
        print(f"Moving to phi: {phi_angle}°, theta: {theta_angle}°")
        # Start angles are the current angles of the servos
        current_phi = phi_servo.angle if phi_servo.angle is not None else 0
        current_theta = theta_servo.angle if theta_servo.angle is not None else 0
        # Smooth movement over 1000 ms
        smooth_move_servo(phi_servo, current_phi, phi_angle, 1)
        smooth_move_servo(theta_servo, current_theta, theta_angle, 1)
        sleep(pause)  # Hold the position for the specified timeout

if __name__ == "__main__":
    try:
        move_servos(movements)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        phi_servo.detach()
        theta_servo.detach()
        print("Final servos detached and program ended.")
