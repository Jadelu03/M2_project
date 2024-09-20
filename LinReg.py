import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


df = pd.read_csv('SENSORY_mod.csv')

# Define your variables
x = df['Melting']
y = df['Crispy']

# Add a constant to the independent variable for the intercept
X = sm.add_constant(x)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the regression line parameters
slope = model.params[1]
intercept = model.params[0]
r_squared = model.rsquared

# Create the plot
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, scatter_kws={'s':50}, line_kws={'color': 'red'}, ci=None)

# Add annotations for R^2 value and slope
plt.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$\nSlope = {slope:.2f}', 
         ha='left', va='top', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Set plot labels and title
plt.xlabel('Melting')
plt.ylabel('Crispy')
plt.title('Melting vs. Crispy')

# Show plot
plt.show()