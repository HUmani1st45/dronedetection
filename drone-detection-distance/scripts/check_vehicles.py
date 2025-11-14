import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

vehicles = client.listVehicles()
print("✅ Available vehicles in the simulator:", vehicles)

# Try enabling control of Observer
try:
    client.enableApiControl(True, vehicle_name="Observer")
    print("✅ Observer is controllable via API.")
except Exception as e:
    print("❌ Could not control 'Observer':", e)
