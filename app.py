import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 🚀 Swapped algorithm
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
print("🔄 Loading Titanic training dataset...")
train_df = pd.read_csv('titanic/train.csv')

# 2. Handle Missing Data
print("🛠️ Cleaning missing values...")
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df = train_df.drop(['Cabin'], axis=1)

# 3. Feature Engineering & Categorical Encoding
print("📊 Engineering features and encoding categories...")
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# Let's also add a 'FarePerPerson' feature to give the model better wealth context
train_df['FarePerPerson'] = train_df['Fare'] / train_df['FamilySize']

train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 4. Separate Features and Target
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Upgraded Model
print("🤖 Training the Random Forest Classifier...")
# n_estimators=100 means we are building 100 individual decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
model.fit(X_train, y_train)

# 6. Evaluate
print("\n📈 Model Evaluation Results:")
y_pred = model.predict(X_test)
print(f"New Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))