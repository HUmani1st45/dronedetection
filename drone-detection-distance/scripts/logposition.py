import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

# change name if you want the other drone
name = "Drone1"

pose = client.simGetVehiclePose(vehicle_name=name)
p = pose.position
o = pose.orientation

print(f"{name} position:")
print(f"  x = {p.x_val:.2f}  y = {p.y_val:.2f}  z = {p.z_val:.2f}")
print(f"{name} orientation (quaternion):")
print(f"  x = {o.x_val:.4f}  y = {o.y_val:.4f}  z = {o.z_val:.4f}  w = {o.w_val:.4f}")
