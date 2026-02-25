
"""A clone of 'sensors' utility on Linux printing hardware temperatures.

$ python3 scripts/sensors.py
asus
    asus                 47.0 °C (high = None °C, critical = None °C)

acpitz
    acpitz               47.0 °C (high = 103.0 °C, critical = 103.0 °C)

coretemp
    Physical id 0        54.0 °C (high = 100.0 °C, critical = 100.0 °C)
    Core 0               47.0 °C (high = 100.0 °C, critical = 100.0 °C)
    Core 1               48.0 °C (high = 100.0 °C, critical = 100.0 °C)
    Core 2               47.0 °C (high = 100.0 °C, critical = 100.0 °C)
    Core 3               54.0 °C (high = 100.0 °C, critical = 100.0 °C)
"""

import sys

import psutil


def main():
    if not hasattr(psutil, "sensors_temperatures"):
        sys.exit("platform not supported")
    temps = psutil.sensors_temperatures()
    if not temps:
        sys.exit("can't read any temperature")
    for name, entries in temps.items():
        print(name)
        for entry in entries:
            print(f"{entry.current} °C")
        print()


if __name__ == '__main__':
    main()