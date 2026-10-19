from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load the datasets
iris = load_iris()
x, y = iris.data, iris.target
# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

# train the model
clf.fit(x_train, y_train)

# save the model to a file
joblib.dump(clf, 'iris_model.pkl')
print("Model trained and saved as 'iris_model.pkl'")
