import pandas as pd
from Lesson import RegressionModel  # import your class from Lesson.ipynb

# Load the dataset used by the unit test
data = pd.read_csv("tests/files/assignment8Data.csv")
x = data[['sex', 'age', 'educ', 'white']]
y = data['incwage']

# Run your regression
reg = RegressionModel(x, y, create_intercept=True)
reg.ols_regression()

# Display your computed values
print(pd.DataFrame(reg.results).T)