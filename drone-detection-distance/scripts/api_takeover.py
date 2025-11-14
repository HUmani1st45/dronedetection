import airsim, time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# This command makes it hold position (zero velocity)
client.hoverAsync().join()

print("Drone is now holding position.")
time.sleep(10)  # take pictures, etc.

# Give control back to the joystick
client.enableApiControl(False)
