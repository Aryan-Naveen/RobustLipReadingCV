import numpy as np

import matplotlib.pyplot as plt

num_iterations = [0, 1, 2,3, 4]
cer = [1.333, 1.0667, 0.8, 1.4, 1.6]
wer = [1.67, 1.333, 1.333, 2 ,2]

fig, ax = plt.subplots(1, 1)

ax.plot(num_iterations, cer, color='b', label='CER')
ax.scatter(num_iterations, cer, color='b')
ax.plot(num_iterations, wer, color = 'orange', label = 'WER')
ax.scatter(num_iterations, wer, color='orange')
ax.set_ylabel('Error Rates')
ax.set_xlabel('Number of Deblurring Iterations')
ax.axvline(x = 2, color = 'r', linestyle= 'dashed', label = 'Optimal Iteration Count')
ax.legend()
plt.show()
