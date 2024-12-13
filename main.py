import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack

# Подавление предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)

# Загрузка данных
data = pd.read_csv("data.csv")

# Удаление дубликатов
data = data.drop_duplicates()
print(f"Data shape after removing duplicates: {data.shape}")

# Проверка на балансировку классов
print("Class distribution:")
print(data['toxic'].value_counts())

# Визуализация распределения классов
plt.figure(figsize=(6, 4))
sns.countplot(data['toxic'], palette='viridis')
plt.title("Распределение классов (токсичные/нетоксичные)")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.show()

# Проверка корреляции признаков с целевой переменной
print("\nCorrelation with target variable:")
numeric_data = data.select_dtypes(include=[np.number])  # Оставляем только числовые столбцы
correlation = numeric_data.corr()['toxic']
print(correlation)

# Удаление признаков с высокой корреляцией с целевой переменной
high_corr_features = correlation[abs(correlation) > 0.9].index.tolist()
high_corr_features.remove('toxic')  # Не удаляем целевую переменную
print(f"\nRemoving highly correlated features: {high_corr_features}")
data = data.drop(columns=high_corr_features)

# Разделение данных
X_text = data['comment_text']
X_numeric = data.select_dtypes(include=[np.number]).drop(columns=['toxic'])  # Числовые признаки
y = data['toxic']

# Добавление шума к числовым данным для усложнения задачи
X_numeric += np.random.normal(0, 0.1, X_numeric.shape)

# Разделение данных на обучающую и тестовую выборки
X_train_text, X_test_text, X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(
    X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF для текстовых данных (обучение только на обучающих данных)
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf.fit(X_train_text)  # Обучение на обучающих данных
X_train_tfidf = tfidf.transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Масштабирование числовых данных
scaler = StandardScaler()
scaler.fit(X_train_numeric)  # Обучение на обучающих данных
X_train_numeric_scaled = scaler.transform(X_train_numeric)
X_test_numeric_scaled = scaler.transform(X_test_numeric)

# Объединение текстовых и числовых данных
X_train = hstack([X_train_tfidf, X_train_numeric_scaled])
X_test = hstack([X_test_tfidf, X_test_numeric_scaled])

# Функция для обучения модели и вывода метрик, включая матрицы ошибок
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Визуализация матрицы ошибок
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='viridis')
    plt.title(f"Матрица ошибок для модели: {model_name}")
    plt.show()

    return model.score(X_test, y_test)

# Словарь для хранения точности моделей
model_scores = {}

# 1. Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)
model_scores['Random Forest'] = evaluate_model(rf_model, "Random Forest")

# 2. Логистическая регрессия
lr_model = LogisticRegression(max_iter=1000, random_state=42)
model_scores['Logistic Regression'] = evaluate_model(lr_model, "Logistic Regression")

# 3. Метод опорных векторов (SVM)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
model_scores['SVM'] = evaluate_model(svm_model, "Support Vector Machine")

# 4. Наивный байесовский классификатор
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)  # Используем только текстовые данные для Naive Bayes
y_pred_nb = nb_model.predict(X_test_tfidf)
print("=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))
cm_nb = confusion_matrix(y_test, y_pred_nb)
ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=nb_model.classes_).plot(cmap='viridis')
plt.title("Матрица ошибок для модели: Naive Bayes")
plt.show()

# 5. LightGBM
lgbm_model = LGBMClassifier(random_state=42, n_estimators=50, max_depth=10, min_child_samples=5, learning_rate=0.1)
model_scores['LightGBM'] = evaluate_model(lgbm_model, "LightGBM")

# Итоговые результаты
print("=== Итоговые результаты ===")
for model_name, score in model_scores.items():
    print(f"{model_name}: {score:.4f}")