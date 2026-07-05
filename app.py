import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
print("🔄 Loading Titanic training dataset...")
# FIXED: Pointing correctly to the 'titanic' subfolder where your file actually sits
train_df = pd.read_csv('titanic/train.csv')

# 2. Handle Missing Data
print("🛠️ Cleaning missing values...")
# Fill missing Age values with the median age of all passengers
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
# Fill missing Embarked values with the most frequent port (mode)
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
# Drop the Cabin column due to excessive missing data
train_df = train_df.drop(['Cabin'], axis=1)

# 3. Feature Engineering & Categorical Encoding
print("📊 Engineering features and encoding categories...")
# Convert Sex into binary numerical values (0 for male, 1 for female)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
# Convert Embarked ports into dummy indicator columns
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
# Combine siblings, spouses, parents, and children to create a total FamilySize feature
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# Drop identifiers that do not contribute to survival patterns
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 4. Separate Features and Target
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
print("🤖 Training the Logistic Regression baseline model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Evaluate
print("\n📈 Model Evaluation Results:")
y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))