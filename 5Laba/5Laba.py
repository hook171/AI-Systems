from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import randint
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Поиск по решетке (Grid Search)
print("=== Grid Search ===")
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Лучшие параметры (Grid Search):", grid_search.best_params_)
print("Точность (Grid Search):", grid_search.best_score_)

# 2. Случайный поиск (Random Search)
print("\n=== Random Search ===")
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("Лучшие параметры (Random Search):", random_search.best_params_)
print("Точность (Random Search):", random_search.best_score_)

# 3. Байесовский поиск (Bayesian Optimization)
print("\n=== Bayesian Optimization ===")
param_bayes = {
    'n_estimators': Integer(10, 200),
    'max_depth': Categorical([None, 10, 20, 30]),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4)
}

model = RandomForestClassifier(random_state=42)
bayes_search = BayesSearchCV(estimator=model, search_spaces=param_bayes, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
bayes_search.fit(X_train, y_train)

print("Лучшие параметры (Bayesian Optimization):", bayes_search.best_params_)
print("Точность (Bayesian Optimization):", bayes_search.best_score_)

# Сравнение результатов на тестовой выборке
print("\n=== Сравнение результатов на тестовой выборке ===")

# Grid Search
model_grid = RandomForestClassifier(**grid_search.best_params_, random_state=42)
model_grid.fit(X_train, y_train)
y_pred_grid = model_grid.predict(X_test)
print("Точность Grid Search на тестовой выборке:", accuracy_score(y_test, y_pred_grid))

# Random Search
model_random = RandomForestClassifier(**random_search.best_params_, random_state=42)
model_random.fit(X_train, y_train)
y_pred_random = model_random.predict(X_test)
print("Точность Random Search на тестовой выборке:", accuracy_score(y_test, y_pred_random))

# Bayesian Optimization
model_bayes = RandomForestClassifier(**bayes_search.best_params_, random_state=42)
model_bayes.fit(X_train, y_train)
y_pred_bayes = model_bayes.predict(X_test)
print("Точность Bayesian Optimization на тестовой выборке:", accuracy_score(y_test, y_pred_bayes))

# Матрицы ошибок и отчеты для всех методов
def evaluate_model(y_true, y_pred, method_name):
    print(f"\nМатрица ошибок ({method_name}):")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nОтчет по классификации ({method_name}):")
    print(classification_report(y_true, y_pred, target_names=data.target_names))

# Grid Search
evaluate_model(y_test, y_pred_grid, "Grid Search")

# Random Search
evaluate_model(y_test, y_pred_random, "Random Search")

# Bayesian Optimization
evaluate_model(y_test, y_pred_bayes, "Bayesian Optimization")