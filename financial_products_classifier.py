import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic dataset
np.random.seed(42)
n_samples = 500

# Features: volatility, returns, liquidity
volatility = np.random.rand(n_samples)
returns = np.random.randn(n_samples) * 0.02  # Simulating small daily returns
liquidity = np.random.randint(1, 100, size=n_samples)  # Arbitrary liquidity measure

# Labels: 0 = Equities, 1 = Bonds, 2 = Currencies
labels = np.random.choice([0, 1, 2], size=n_samples)

# Create DataFrame
data = pd.DataFrame({
    'volatility': volatility,
    'returns': returns,
    'liquidity': liquidity,
    'label': labels
})

# Split dataset into train and test sets
X = data[['volatility', 'returns', 'liquidity']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=500)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=['Equities', 'Bonds', 'Currencies']))
