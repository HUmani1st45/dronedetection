import airsim, time

client = airsim.MultirotorClient()
client.confirmConnection()

# Try to discover available vehicles (newer AirSim)
veh = "Drone1"
try:
    names = client.listVehicles()
    if names:
        veh = names[0]
        print("Discovered vehicles:", names, "-> using", veh)
    else:
        print("No vehicles reported by API, will try 'Drone1' then default ''.")
except AttributeError:
    # Older API: listVehicles may not exist
    pass

def try_vehicle(name):
    try:
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)
        client.takeoffAsync(vehicle_name=name).join()
        time.sleep(1)
        client.moveToPositionAsync(10, 0, -10, 5, vehicle_name=name).join()
        client.landAsync(vehicle_name=name).join()
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
        print(f"SUCCESS with vehicle_name='{name}'")
        return True
    except Exception as e:
        print(f"FAILED with vehicle_name='{name}':", e)
        return False

# Try with discovered/'Drone1', then with default vehicle (empty string)
if not try_vehicle(veh):
    try_vehicle("")  # some builds use the default vehicle with empty name
