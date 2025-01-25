# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Загрузка данных
train_data = pd.read_csv('C:/Users/ARTEM/Documents/GitHub/AI-Systems/4Laba/Spaceship_titanic/train.csv')
test_data = pd.read_csv('C:/Users/ARTEM/Documents/GitHub/AI-Systems/4Laba/Spaceship_titanic/test.csv')

# Предварительная обработка данных
# Удаление малозначащих данных
train_data.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name'], axis=1, inplace=True)

# Обработка категориальных признаков и числовых данных
categorical_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Создание трансформеров для числовых и категориальных данных
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Замена пропущенных значений медианой
    ('scaler', StandardScaler())])  # Нормализация данных

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Замена пропущенных значений наиболее частым значением
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # Кодирование категориальных признаков

# Объединение трансформеров
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Применение к числовым признакам
        ('cat', categorical_transformer, categorical_features)])  # Применение к категориальным признакам

# Отделение целевой функции от датасета
X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели Random Forest
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])
rf_model.fit(X_train, y_train)

# Обучение модели XGBoost
xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])  # Удален use_label_encoder
xgb_model.fit(X_train, y_train)

# Оценка моделей
# Предсказание и оценка для Random Forest
rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy: ", accuracy_score(y_test, rf_predictions))

# Предсказание и оценка для XGBoost
xgb_predictions = xgb_model.predict(X_test)
print("XGBoost Accuracy: ", accuracy_score(y_test, xgb_predictions))