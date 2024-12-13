import pandas as pd
import random
import numpy as np

# Примеры нетоксичных комментариев
non_toxic_comments = [
    "Это замечательный комментарий!",
    "Спасибо за вашу помощь!",
    "Очень полезная информация.",
    "Мне понравился ваш пост.",
    "Прекрасный день, чтобы обсудить эту тему.",
    "Вы отлично справились, продолжайте в том же духе!",
    "Это была интересная статья, спасибо.",
    "Ваш комментарий очень поддерживает.",
    "Мне нравится, как вы объясняете.",
    "Это действительно хороший совет."
]

# Примеры токсичных комментариев
toxic_comments = [
    "Ты ничего не понимаешь!",
    "Какой ужасный комментарий.",
    "Это просто бред.",
    "Ты совершенно некомпетентен.",
    "Зачем ты вообще пишешь?",
    "Ужасный пост, удали это.",
    "Ты только тратишь время людей.",
    "Никогда не читал ничего хуже.",
    "Твои слова — это полный провал.",
    "Какой позорный текст!"
]

# Функция для добавления случайного шума в текст
def add_noise_to_comment(comment):
    noise_options = ["!", ".", "?", "слово", "пример", "ТЕКСТ"]
    if random.random() > 0.7:
        comment += " " + random.choice(noise_options)
    if random.random() > 0.7:
        comment = comment.upper() if random.random() > 0.5 else comment.lower()
    return comment

# Генерация данных
data = {
    "comment_text": [],
    "feature_1": [],
    "feature_2": [],
    "feature_3": [],
    "feature_4": [],
    "feature_5": [],
    "feature_6": [],
    "feature_7": [],
    "feature_8": [],
    "random_feature_1": [],  # Случайный независимый признак
    "random_feature_2": [],  # Случайный независимый признак
    "toxic": []
}

# Генерация нетоксичных комментариев
for _ in range(500):
    comment = add_noise_to_comment(random.choice(non_toxic_comments))
    data["comment_text"].append(comment)
    data["feature_1"].append(len(comment) + random.randint(-2, 2))  # Длина комментария с шумом
    data["feature_2"].append(len(comment.split()) + random.randint(-1, 1))  # Количество слов с шумом
    data["feature_3"].append(random.randint(0, 3))  # Случайное количество восклицательных знаков
    data["feature_4"].append(random.randint(0, 2))  # Случайное количество вопросов
    data["feature_5"].append(random.randint(0, 5))  # Случайный числовой признак
    data["feature_6"].append(random.random())  # Случайное число (0-1)
    data["feature_7"].append(len(set(comment)) + random.randint(-2, 2))  # Количество уникальных символов с шумом
    data["feature_8"].append(random.randint(0, 1))  # Независимый случайный признак
    data["random_feature_1"].append(random.uniform(0, 1))  # Случайный непрерывный признак
    data["random_feature_2"].append(random.randint(0, 10))  # Случайный дискретный признак
    data["toxic"].append(0)

# Генерация токсичных комментариев
for _ in range(500):
    comment = add_noise_to_comment(random.choice(toxic_comments))
    data["comment_text"].append(comment)
    data["feature_1"].append(len(comment) + random.randint(-2, 2))  # Длина комментария с шумом
    data["feature_2"].append(len(comment.split()) + random.randint(-1, 1))  # Количество слов с шумом
    data["feature_3"].append(random.randint(2, 5))  # Случайное количество восклицательных знаков
    data["feature_4"].append(random.randint(1, 3))  # Случайное количество вопросов
    data["feature_5"].append(random.randint(0, 5))  # Случайный числовой признак
    data["feature_6"].append(random.random())  # Случайное число (0-1)
    data["feature_7"].append(len(set(comment)) + random.randint(-2, 2))  # Количество уникальных символов с шумом
    data["feature_8"].append(random.randint(0, 1))  # Независимый случайный признак
    data["random_feature_1"].append(random.uniform(0, 1))  # Случайный непрерывный признак
    data["random_feature_2"].append(random.randint(0, 10))  # Случайный дискретный признак
    data["toxic"].append(1)

# Перемешивание данных
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Проверка корреляции (только для числовых столбцов)
numerical_df = df.select_dtypes(include=[np.number])  # Выбираем только числовые столбцы
correlation = numerical_df.corr()  # Вычисляем корреляцию только для числовых данных
high_corr_features = correlation["toxic"][abs(correlation["toxic"]) > 0.8].index.tolist()
high_corr_features.remove("toxic")
print(f"Удаляем признаки с высокой корреляцией: {high_corr_features}")

# Удаление высоко коррелированных признаков
df = df.drop(columns=high_corr_features)

# Сохранение в CSV
df.to_csv("data.csv", index=False)
print("Файл data.csv успешно создан.")