import RPi.GPIO as GPIO
import time

# GPIO pin numbers for the motor drivers
DIR_X = 20   # Direction GPIO Pin for the X-axis motor
STEP_X = 21  # Step GPIO Pin for the X-axis motor
DIR_Y = 19   # Direction GPIO Pin for the Y-axis motor
STEP_Y = 26  # Step GPIO Pin for the Y-axis motor
CW = 1       # Clockwise Rotation
CCW = 0      # Counterclockwise Rotation

def initialize_motors():
    """
    Initialize the Raspberry Pi GPIO pins for motor control.
    """
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin-numbering scheme
    GPIO.setup(DIR_X, GPIO.OUT)
    GPIO.setup(STEP_X, GPIO.OUT)
    GPIO.setup(DIR_Y, GPIO.OUT)
    GPIO.setup(STEP_Y, GPIO.OUT)

    # Set both motors to a default direction (optional)
    GPIO.output(DIR_X, CW)
    GPIO.output(DIR_Y, CW)

def clean_up():
    """
    Clean up by resetting GPIO resources. It's important to call this function
    before exiting the program to avoid potential damage.
    """
    GPIO.cleanup()

if __name__ == "__main__":
    try:
        initialize_motors()
        print("Motors initialized. Ready to control.")
        # The control code or loop
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        clean_up()
        print("GPIO cleanup done. Exiting program.")

