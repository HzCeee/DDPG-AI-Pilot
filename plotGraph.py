import csv
import matplotlib.pyplot as plt

fileHandle = open('progress.csv')
fileReader = csv.DictReader(fileHandle)

average_success_rate = []
average_distance= []
average_return = []

for row in fileReader:
    average_distance.append(float(row['AI_distance'][:6]))
    if row['AI_success_rate']:
        average_success_rate.append(float(row['AI_success_rate'][:6]))
    average_return.append(float(row['AI_return'][:6]))

plt.figure(1)
plt.plot(average_distance)
plt.xlabel('episodes/2000')
plt.ylabel('m average distance')

plt.figure(2)
plt.plot(average_success_rate)
plt.xlabel('episodes/2000')
plt.ylabel('% success_rate')

plt.figure(3)
plt.plot(average_return)
plt.xlabel('episodes/2000')
plt.ylabel('average return')

plt.show()
