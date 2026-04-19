import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

thermocouple_locations = [
    (0,0),
    (-0.5, -0.25),
    (0.45, 0.35),
    (0.25, -0.45),
]
dataset = pd.read_csv('training_data/dual_gaussian_4.csv')

plt.figure(figsize=(5, 5))

for i in range(len(thermocouple_locations)):
    point_data = dataset[dataset['sensor_id'] == i]
    time = point_data['time']
    temp = point_data['temperature']
    location = f"({thermocouple_locations[i][0]}, {thermocouple_locations[i][1]})"
    plt.plot(time, temp, label=location)
plt.xlabel('Time')
plt.xticks(np.arange(0, 5.1, 1))
plt.ylabel('Temperature')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Temperature vs Time for 4 Sensors')
plt.legend()
plt.grid()
plt.savefig('report/plots/dual_gaussian_4.png', dpi=300)
plt.show()
