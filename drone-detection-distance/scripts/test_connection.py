import airsim
import time

client = airsim.MultirotorClient()

print("🔌 Trying to connect to AirSim...")

# Try for up to 10 seconds
for attempt in range(20):
    try:
        client.confirmConnection()
        print("✅ Connected to AirSim.")
        break
    except Exception:
        print(f"⏳ Attempt {attempt+1}/20... waiting for simulator.")
        time.sleep(0.5)
else:
    print("❌ Failed to connect to AirSim after 10s.")
    exit()

try:
    vehicles = client.listVehicles()
    print("✅ Vehicles:", vehicles)
except Exception as e:
    print("❌ Could not get vehicle list:", e)
