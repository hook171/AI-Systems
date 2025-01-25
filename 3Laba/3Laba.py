import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error

# Загрузка данных
train_data = pd.read_csv('C:/Users/ARTEM/Documents/GitHub/AI-Systems/3Laba/Titanic/train.csv')
test_data = pd.read_csv('C:/Users/ARTEM/Documents/GitHub/AI-Systems/3Laba/Titanic/test.csv')
gender_submission = pd.read_csv('C:/Users/ARTEM/Documents/GitHub/AI-Systems/3Laba/Titanic/gender_submission.csv')

# Объединение данных для удобства обработки
test_data = test_data.merge(gender_submission, on='PassengerId')
data = pd.concat([train_data, test_data], ignore_index=True)

# Экспертный анализ данных
print(data.info())
print(data.describe())

# Удаление малозначащих данных
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Обработка категориальных признаков
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Замена пропущенных значений
imputer = SimpleImputer(strategy='median')
data[['Age', 'Fare']] = imputer.fit_transform(data[['Age', 'Fare']])

# Отделение целевой функции от датасета
X = data.drop('Survived', axis=1)
y = data['Survived']

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение моделей
# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))

# Логистическая регрессия
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic))

# Лассо регрессия
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print("Lasso Regression MSE:", mean_squared_error(y_test, y_pred_lasso))