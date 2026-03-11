import sys
import psutil

        
def get_cpu_temp():
    if not hasattr(psutil, "sensors_temperatures"):
        # platform not supported
        return None
    temps = psutil.sensors_temperatures()
    if not temps:
        return None
    for name, entries in temps.items():
        for entry in entries:
            if name == "cpu_thermal":
                return entry.current
