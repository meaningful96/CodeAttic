# Step 1) Data Loading
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/meaningful96/CodeAttic/main/Dataset/1_LinearRegression_dataset.txt", sep ="\s+")

xxRaw = dfLoad['xx']
yyRaw = dfLoad['yy']

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Scatter plot of raw data
ax1.plot(xxRaw, yyRaw, 'ro')
ax1.set_title('Scatter plot of raw data')
ax1.set_xlabel('xx')
ax1.set_ylabel('yy')

# Step 2) Analytical Way, wOLS
Ndata = len(xxRaw)
xxRawNP = np.array(xxRaw)
yyRawNP = np.array(yyRaw)
X = np.column_stack([np.ones([Ndata, 1]), xxRaw])

wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yyRaw)

# Step 3) Prediction
xPredict = np.linspace(0, 2, 101)
xPredictPadding = np.column_stack([np.ones([101, 1]), xPredict])

yPredict = wOLS.T.dot(xPredictPadding.T)

# Step 4) Numerical Way, Gradient Descent
eta = 0.1
niterations = 20
wGD = np.zeros([2, 1])

for iteration in range(niterations):
    gradients = -(2 / Ndata) * (X.T.dot(yyRawNP.reshape(Ndata, 1) - X.dot(wGD)))
    wGD = wGD - eta * gradients
    yGD = wGD.T.dot(xPredictPadding.T)
    ax2.plot(xPredict, yGD.flatten(), 'b*', alpha=0.3)

ax2.plot(xxRaw, yyRaw, 'ro')
# Plot the final gradient descent prediction
ax2.plot(xPredict, yGD.flatten(), 'g*', label='GD Final Prediction')

ax2.set_title('Gradient Descent Prediction')
ax2.set_xlabel('xPredict')
ax2.set_ylabel('yGD')

# Show the plots
plt.tight_layout()
plt.show()
