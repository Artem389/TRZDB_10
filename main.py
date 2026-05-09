import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import warnings
import re
import os
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer  # Превращает текст в числа (мешок слов)
from sklearn.model_selection import train_test_split  # Делит данные на обучение и тест
from sklearn.linear_model import LogisticRegression  # Простая модель для классификации
from sklearn.ensemble import RandomForestClassifier  # Лес из деревьев решений
from xgboost import XGBClassifier  # Градиентный бустинг (самая мощная из трёх)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# accuracy_score — доля правильных ответов
# precision_score — сколько из предсказанных позитивных реально позитивные
# recall_score — сколько реальных позитивных мы нашли
# f1_score — среднее гармоническое precision и recall
# confusion_matrix — матрица ошибок (сколько раз попутали позитив с негативом)

from sklearn.metrics.pairwise import cosine_similarity  # Косинусное сходство (чисто математика)

# Визуализация
import matplotlib.pyplot as plt  # Основная библиотека для графиков
from wordcloud import WordCloud  # Красивое облако слов
import matplotlib

matplotlib.use('Agg')  # Говорим matplotlib работать без окна (для сервера)

# Веб-интерфейс
import streamlit as st  # Библиотека для быстрого создания веб-приложений
from streamlit_cookies_manager import EncryptedCookieManager  # Для хранения user_id в куках


# ======================================================================
# ФУНКЦИЯ-ЗАГЛУШКА ДЛЯ ПУСТОГО ОТЧЁТА
# ======================================================================

def generate_empty_report():
    """
    Если отчёт ещё не сгенерирован, создаём картинку-заглушку с надписью.
    Типичный джунский костыль: вместо нормальной инициализации просто проверяем файл.
    """
    if not os.path.exists('student_report.png'):  # Проверяем, есть ли файл (os.path.exists — проверка существования)
        fig, ax = plt.subplots(figsize=(10, 6))  # Создаём пустой холст 10x6 дюймов
        ax.text(0.5, 0.5, 'Отчет еще не сгенерирован\nНажмите кнопку ниже',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        # ax.text — добавляем текст на холст
        # ha='center', va='center' — выравнивание по центру
        # transform=ax.transAxes — координаты (0.5, 0.5) это процент от размера (50% по ширине, 50% по высоте)
        ax.set_title('STUDENT REPORT')  # Заголовок графика
        plt.savefig('student_report.png', dpi=150, bbox_inches='tight')
        # savefig — сохраняем в файл PNG с качеством 150 dpi (точек на дюйм)
        # bbox_inches='tight' — обрезаем лишние поля
        plt.close()  # Закрываем график, чтобы не копился в памяти


# ======================================================================
# 1. СОЗДАНИЕ ДАТАСЕТА (250 отзывов)
# ======================================================================

def create_my_dataset():
    """
    Генерирует синтетические отзывы.
    """
    np.random.seed(42)  # Фиксируем сид (начальное число рандома) — чтобы данные всегда были одинаковыми

    # Списки слов для генерации тональности
    positive_words = [
        "отличный", "прекрасный", "великолепный", "потрясающий", "качественный",
        "надежный", "стильный", "удобный", "быстрый", "красивый",
        "супер", "рекомендую", "доволен", "нравится", "лучший",
        "шикарный", "огонь", "топ", "бомба", "идеальный"
    ]

    negative_words = [
        "ужасный", "плохой", "отвратительный", "разочарован", "кошмар",
        "брак", "сломался", "глючит", "тормозит", "деньги на ветер",
        "не советую", "не покупайте", "фигня", "ерунда", "отстой",
        "хлам", "мусор", "никакой", "бесполезный", "ужас"
    ]

    # Нейтральные шаблоны (для 3 звёзд)
    neutral_templates = [
        "нормальный товар за свои деньги",
        "среднее качество, но пойдет",
        "обычный товар, ничего особенного",
        "на троечку",
        "не плохо и не хорошо",
        "среднячок",
        "можно брать",
        "на один раз",
        "ожидал большего, но сойдет",
        "в целом неплохо"
    ]

    # Список товаров (10 штук, чтобы было разнообразие)
    products = ["Смартфон X", "Ноутбук Pro", "Наушники Bass", "Часы Smart", "Планшет Lite",
                "Колонка BT", "Мышь Wireless", "Клавиатура Mech", "Монитор 4K", "Камера HD"]

    data = []  # Пустой список для будущих записей

    for i in range(250):  # Генерируем 250 записей
        # Выбираем рейтинг со смещением в сторону положительных (p — вероятности для каждого рейтинга)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.12, 0.2, 0.3, 0.3])
        # 8% шанс на 1 звезду, 12% на 2 звезды, 20% на 3, 30% на 4, 30% на 5

        product = np.random.choice(products)  # Случайный товар

        # ЛОГИКА ГЕНЕРАЦИИ ТЕКСТА В ЗАВИСИМОСТИ ОТ РЕЙТИНГА
        if rating >= 4:
            # Если оценка 4-5 — генерируем позитивный текст
            text = f"{np.random.choice(positive_words)} {product.lower()}, {np.random.choice(positive_words)}!"
            sentiment = 1  # 1 = позитив
        elif rating <= 2:
            # Если оценка 1-2 — генерируем негативный текст
            text = f"{np.random.choice(negative_words)} {product.lower()}, {np.random.choice(negative_words)}."
            sentiment = 0  # 0 = негатив
        else:
            # Оценка 3 — нейтральный текст
            text = np.random.choice(neutral_templates) + f" {product.lower()}"
            # Вот тут проблема: определяет sentiment по тексту, а не по rating
            sentiment = 0 if "ожидал" in text or "троечку" in text else 1
            # Если в тексте есть слова "ожидал" или "троечку" — считаем негативом (очень условно)

        # Добавляем запись в список
        data.append({
            'id': i,
            'text': text,
            'rating': rating,  # Оценка пользователя (1-5)
            'sentiment': sentiment,  # Тональность (0=негатив, 1=позитив)
            'date': datetime.now() - timedelta(days=np.random.randint(0, 90)),
            # datetime.now() — текущее время, timedelta — разница во времени
            # np.random.randint(0, 90) — случайное кол-во дней назад (0-89)
            'product': product,
            'user_name': f"Пользователь_{np.random.randint(1, 41)}",  # Имена от 1 до 40
            'user_id': np.random.randint(1, 41)  # ID от 1 до 40
        })

    return pd.DataFrame(data)  # Превращаем список словарей в таблицу DataFrame


# ======================================================================
# 2. ФУНКЦИИ ДЛЯ КЛАССИФИКАЦИИ (Оценка 3)
# ======================================================================

def classify_text(text, model, vectorizer):
    """
    Предсказывает тональность текста и возвращает уверенность модели.

    Параметры:
    - text: строка с отзывом
    - model: обученная модель (LogisticRegression)
    - vectorizer: векторизатор (TfidfVectorizer), превращает текст в числовой вектор

    Возвращает:
    - result: "ПОЗИТИВ" или "НЕГАТИВ"
    - confidence: процент уверенности (от 0 до 1)
    """
    X = vectorizer.transform([text])  # Превращаем текст в массив чисел (TF-IDF)
    # transform — применяет уже обученный векторизатор к новому тексту
    # [text] — передаём список из одного элемента

    pred = model.predict(X)[0]  # Предсказываем класс (0 или 1)
    # predict — метод модели для предсказания
    # [0] — берём первый (и единственный) элемент результата

    proba = model.predict_proba(X)[0]  # Получаем вероятности для ВСЕХ классов
    # predict_proba возвращает [вероятность_негатива, вероятность_позитива]
    # [0] — берём первую строку (у нас один текст)

    confidence = proba[int(pred)]  # Берём вероятность предсказанного класса
    # int(pred) — преобразуем 0/1 в индекс для массива proba

    result = "ПОЗИТИВ" if pred == 1 else "НЕГАТИВ"  # Тернарный оператор (сокращённый if)
    return result, confidence


def top_products_by_positive(df, top_n=5):
    """
    Находит топ-N товаров с самой большой долей позитивных отзывов.

    Параметры:
    - df: датафрейм с отзывами
    - top_n: сколько товаров вернуть (по умолчанию 5)

    Возвращает:
    - список словарей [{product, positive_ratio, total}, ...]
    """
    result = []
    for product in df['product'].unique():  # unique() — список уникальных товаров
        product_data = df[df['product'] == product]  # Фильтруем только отзывы на этот товар
        if len(product_data) > 0:  # Если есть хоть один отзыв
            positive_count = len(product_data[product_data['sentiment'] == 1])  # Считаем позитивные
            total = len(product_data)  # Всего отзывов
            ratio = positive_count / total  # Доля позитивных (от 0 до 1)
            result.append({'product': product, 'positive_ratio': ratio, 'total': total})

    # Сортируем по убыванию доли позитива
    result_sorted = sorted(result, key=lambda x: x['positive_ratio'], reverse=True)
    # key=lambda x: x['positive_ratio'] — сортируем по ключу 'positive_ratio'
    # reverse=True — от большего к меньшему

    return result_sorted[:top_n]  # Возвращаем только первые top_n


def top_negative_words(df, top_n=5):
    """
    Находит самые частые слова в негативных отзывах.
    Джун даже не удосужился использовать нормальную токенизацию — просто регулярки.

    Параметры:
    - df: датафрейм с отзывами
    - top_n: сколько слов вернуть

    Возвращает:
    - список кортежей [(слово, частота), ...]
    """
    negative_texts = df[df['sentiment'] == 0]['text']  # Только негативные отзывы
    all_words = []

    for text in negative_texts:  # Проходим по каждому отзыву
        words = re.findall(r'\b[а-яё]+\b', text.lower())  # Ищем все русские слова
        # \b — граница слова, [а-яё]+ — одна или больше русских букв
        # text.lower() — приводим к нижнему регистру
        all_words.extend(words)  # Добавляем найденные слова в общий список

    # Стоп-слова
    stop_words = ['и', 'в', 'не', 'на', 'с', 'по', 'а', 'но', 'что', 'как', 'это',
                  'из', 'у', 'я', 'за', 'от', 'для']

    filtered = [w for w in all_words if w not in stop_words and len(w) > 2]
    # Убираем стоп-слова и слишком короткие слова (меньше 3 букв)

    word_counts = Counter(filtered)  # Counter — считает частоту каждого слова (как словарь)
    return word_counts.most_common(top_n)  # most_common() — возвращает самые частые элементы


# ======================================================================
# 3. БАЗА ДАННЫХ SQLite (чистый SQL в коде)
# ======================================================================

def init_db():
    """
    Инициализирует базу данных: создаёт таблицу reviews, если её нет.
    Джун использует SQLite — лёгкая БД, которая хранится в одном файле.
    """
    conn = sqlite3.connect('reviews.db')  # Подключаемся к файлу reviews.db (создаётся если нет)
    c = conn.cursor()  # Курсор — объект для выполнения SQL-запросов
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT, -- Автоинкрементный ID
                     user_id
                     TEXT,          -- ID пользователя (текстовый)
                     user_name
                     TEXT,          -- Имя пользователя
                     text
                     TEXT,          -- Текст отзыва
                     rating
                     INTEGER,       -- Оценка (1-5)
                     product
                     TEXT,          -- Название товара
                     sentiment
                     INTEGER,       -- Тональность (0=негатив, 1=позитив)
                     date
                     TEXT
                 )''')

    conn.commit()  # Сохраняем изменения
    conn.close()  # Закрываем соединение


def add_review_to_db(user_id, user_name, text, rating, product, sentiment):
    """
    Добавляет отзыв в базу данных.
    ИСПОЛЬЗУЕТ ПАРАМЕТРИЗОВАННЫЙ ЗАПРОС (защита от SQL-инъекций).
    """
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO reviews (user_id, user_name, text, rating, product, sentiment, date) VALUES (?, ?, ?, ?, ?, ?, ?)',
        # ? — плейсхолдеры, вместо них подставляются значения из кортежа
        (user_id, user_name, text, rating, product, sentiment, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    # strftime — форматирует дату как строку "ГГГГ-ММ-ДД ЧЧ:ММ:СС"
    conn.commit()
    conn.close()


def get_all_reviews():
    """Получает ВСЕ отзывы из базы данных."""
    conn = sqlite3.connect('reviews.db')
    df = pd.read_sql_query('SELECT * FROM reviews', conn)
    # read_sql_query — выполняет SQL и сразу возвращает pandas DataFrame
    conn.close()
    return df


def get_user_reviews(user_id):
    """
    Получает отзывы конкретного пользователя.
    """
    conn = sqlite3.connect('reviews.db')
    df = pd.read_sql_query(f"SELECT * FROM reviews WHERE user_id='{user_id}'", conn)
    conn.close()
    return df


# ======================================================================
# 4. ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ (User-Based Collaborative Filtering)
# ======================================================================

def get_recommendations_for_user(user_id, df_all, top_n=3):
    """
    Рекомендует товары на основе похожих пользователей.

    Алгоритм User-Based CF:
    1. Строим матрицу user-item (пользователи x товары)
    2. Находим косинусное сходство между пользователями
    3. Берём топ-10 самых похожих пользователей
    4. Считаем взвешенный рейтинг для товаров, которые текущий пользователь ещё не оценил
    """

    user_reviews = df_all[df_all['user_id'] == user_id]  # Отзывы текущего пользователя

    if len(user_reviews) == 0:
        # Если пользователь новый (нет отзывов) — рекомендуем самые популярные товары
        popular = df_all.groupby('product')['rating'].mean().sort_values(ascending=False)
        # groupby — группируем по товарам, mean() — средний рейтинг каждого товара
        # sort_values(ascending=False) — сортируем по убыванию
        return list(popular.head(top_n).index)  # Возвращаем названия товаров

    # Строим сводную таблицу: строки = пользователи, столбцы = товары, значения = средний рейтинг
    user_item = df_all.pivot_table(index='user_id', columns='product', values='rating', aggfunc='mean').fillna(0)
    # fillna(0) — заменяем NaN (нет оценки) на 0

    try:
        user_vector = user_item.loc[user_id].values.reshape(1, -1)
        # .loc[user_id] — строка текущего пользователя
        # .values — numpy массив
        # reshape(1, -1) — превращаем в двумерный массив (1 строка, N столбцов) для cosine_similarity
    except:
        # Если пользователя нет в матрице — возвращаем популярное
        return list(df_all.groupby('product')['rating'].mean().sort_values(ascending=False).head(top_n).index)

    # Считаем косинусное сходство между текущим пользователем и всеми остальными
    similarity = cosine_similarity(user_item.values, user_vector).flatten()
    # cosine_similarity возвращает матрицу, flatten() — превращает в одномерный массив

    user_item['similarity'] = similarity  # Добавляем столбец со сходством
    similar_users = user_item.sort_values('similarity', ascending=False).head(10)
    # Сортируем по убыванию сходства, берём топ-10

    similar_users = similar_users[similar_users.index != user_id]
    # Убираем самого пользователя (он и так на 1 месте с similarity=1.0)

    # Считаем взвешенный рейтинг для каждого товара
    products_scores = {}
    for product in df_all['product'].unique():
        if product not in user_reviews['product'].values:  # Только товары, которые пользователь не оценивал
            scores = []
            for sim_user in similar_users.index:  # Проходим по похожим пользователям
                if product in similar_users.columns:
                    val = similar_users.loc[sim_user, product]  # Оценка похожего пользователя
                    sim = similar_users.loc[sim_user, 'similarity']  # Сходство с ним
                    if val > 0:  # Если оценка положительная
                        scores.append(val * sim)  # Взвешенная оценка (оценка * сходство)
            if scores:
                products_scores[product] = np.mean(scores)  # Средняя взвешенная оценка

    sorted_products = sorted(products_scores.items(), key=lambda x: x[1], reverse=True)
    # items() — список кортежей (товар, рейтинг), сортируем по рейтингу

    return [p[0] for p in sorted_products[:top_n]]  # Возвращаем только названия товаров


# ======================================================================
# 5. СРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ
# ======================================================================

def compare_models(X_train, X_test, y_train, y_test):
    """
    Обучает и сравнивает 3 модели:
    - Logistic Regression (простая линейная модель)
    - Random Forest (ансамбль деревьев решений)
    - XGBoost (градиентный бустинг — самая мощная)

    Возвращает:
    - DataFrame с метриками
    - лучшую модель (LogisticRegression) для дальнейшего использования
    """
    results = []

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000)
    # max_iter=1000 — максимальное кол-во итераций для сходимости
    lr.fit(X_train, y_train)  # Обучаем модель
    y_pred_lr = lr.predict(X_test)  # Предсказываем на тестовых данных
    results.append({
        'Модель': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_lr),  # Доля правильных ответов
        'Precision': precision_score(y_test, y_pred_lr, zero_division=0),  # Точность позитивных
        'Recall': recall_score(y_test, y_pred_lr, zero_division=0),  # Полнота позитивных
        'F1': f1_score(y_test, y_pred_lr, zero_division=0)  # F1-мера
        # zero_division=0 — что вернуть если деление на 0 (нет позитивных предсказаний)
    })

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # n_estimators=100 — 100 деревьев в лесу
    # random_state=42 — фиксируем сид для воспроизводимости
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append({
        'Модель': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'F1': f1_score(y_test, y_pred_rf, zero_division=0)
    })

    # --- XGBoost ---
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    # use_label_encoder=False — отключаем устаревший кодировщик меток
    # eval_metric='logloss' — метрика для оценки в процессе обучения
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append({
        'Модель': 'XGBoost',
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'Precision': precision_score(y_test, y_pred_xgb, zero_division=0),
        'Recall': recall_score(y_test, y_pred_xgb, zero_division=0),
        'F1': f1_score(y_test, y_pred_xgb, zero_division=0)
    })

    return pd.DataFrame(results), lr  # Возвращаем lr как "основную" модель


# ======================================================================
# 6. ГЕНЕРАЦИЯ ОТЧЁТА (ГРАФИКИ + ТЕКСТ)
# ======================================================================

def generate_report_plots(df, df_binary, model, vectorizer):
    """
    Создаёт 6 графиков в одном изображении student_report.png
    и сохраняет текстовый отчёт student_report.txt

    Макет: 2 строки x 3 столбца
    [0,0] Гистограмма оценок
    [0,1] Динамика отзывов по дням
    [0,2] Топ товаров по доле позитива
    [1,0] Матрица ошибок
    [1,1] Средний рейтинг товаров
    [1,2] Облако слов негативных отзывов
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # figsize=(18,12) — размер холста 18x12 дюймов

    fig.suptitle('STUDENT REPORT - Анализ отзывов', fontsize=16, fontweight='bold')

    # [0,0] Гистограмма распределения оценок
    axes[0, 0].hist(df['rating'], bins=5, edgecolor='black', color='royalblue', alpha=0.7)
    # bins=5 — 5 столбцов (по количеству оценок 1-5)
    # edgecolor='black' — чёрная граница столбцов
    # alpha=0.7 — прозрачность 70%
    axes[0, 0].set_title('Распределение оценок')
    axes[0, 0].set_xlabel('Оценка')
    axes[0, 0].set_ylabel('Количество')

    # [0,1] Динамика отзывов по дням
    daily = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    # df['date'].dt.date — берём только дату (без времени)
    # groupby(...).size() — считаем количество отзывов за каждый день
    # reset_index(name='count') — превращаем группировку обратно в DataFrame

    daily['date'] = pd.to_datetime(daily['date'])  # Превращаем обратно в datetime
    daily = daily.sort_values('date')  # Сортируем по дате

    axes[0, 1].plot(daily['date'], daily['count'], 'b-o', linewidth=2, markersize=4)
    # 'b-o' — синяя линия (b=blue) с круглыми маркерами (o)
    axes[0, 1].set_title('Динамика отзывов по дням')
    axes[0, 1].tick_params(axis='x', rotation=45)  # Поворачиваем подписи по оси X на 45°

    # [0,2] Топ товаров по доле позитива
    top_prods = top_products_by_positive(df_binary, top_n=10)
    names = [p['product'][:15] for p in top_prods]  # Обрезаем длинные названия
    ratios = [p['positive_ratio'] * 100 for p in top_prods]  # Переводим в проценты
    colors = ['green' if r > 50 else 'red' for r in ratios]  # Зелёный если >50%, красный если <50%

    axes[0, 2].barh(names[::-1], ratios[::-1], color=colors[::-1])
    # barh — горизонтальные столбцы (horizontal)
    # [::-1] — разворачиваем список (чтобы самый позитивный был сверху)
    axes[0, 2].set_title('Доля позитивных отзывов (%)')
    axes[0, 2].set_xlim(0, 100)  # Ось X от 0 до 100%

    # [1,0] Матрица ошибок (Confusion Matrix)
    X = df_binary['text'].values
    y = df_binary['sentiment'].values
    X_vec = vectorizer.transform(X)  # Векторизуем ВСЕ тексты
    y_pred = model.predict(X_vec)  # Предсказываем для ВСЕХ
    cm = confusion_matrix(y, y_pred)  # Строим матрицу 2x2

    im = axes[1, 0].imshow(cm, cmap='Blues')  # Показываем матрицу цветом (синяя гамма)
    for i in range(2):  # Для каждой строки
        for j in range(2):  # Для каждого столбца
            axes[1, 0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            # Пишем число в центре ячейки

    axes[1, 0].set_title('Матрица ошибок')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Негатив', 'Позитив'])
    axes[1, 0].set_yticklabels(['Негатив', 'Позитив'])

    # [1,1] Средний рейтинг товаров
    avg_rating = df.groupby('product')['rating'].mean().sort_values()
    axes[1, 1].barh(avg_rating.index, avg_rating.values, color='seagreen', alpha=0.7)
    axes[1, 1].set_title('Средний рейтинг товаров')
    axes[1, 1].set_xlim(0, 5)

    # [1,2] Облако слов негативных отзывов
    negative_words = top_negative_words(df_binary, top_n=30)
    if negative_words:
        wordcloud_dict = dict(negative_words)  # Превращаем список кортежей в словарь
        wc = WordCloud(width=400, height=300, background_color='white', max_words=30).generate_from_frequencies(
            wordcloud_dict)
        # WordCloud создаёт изображение где размер слова пропорционален его частоте
        axes[1, 2].imshow(wc, interpolation='bilinear')
        # interpolation='bilinear' — сглаживание пикселей
        axes[1, 2].set_title('Слова в негативных отзывах')
        axes[1, 2].axis('off')  # Убираем оси координат
    else:
        axes[1, 2].text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        axes[1, 2].set_title('Слова в негативных отзывах')

    plt.tight_layout()  # Автоматически подгоняем чтобы графики не накладывались
    plt.savefig('student_report.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- СОХРАНЯЕМ ТЕКСТОВЫЙ ОТЧЁТ ---
    with open('student_report.txt', 'w', encoding='utf-8') as f:  # 'w' — режим записи, перезаписывает файл
        f.write("=" * 50 + "\n")  # Разделитель из 50 знаков =
        f.write("ОТЧЕТ ПО АНАЛИЗУ ОТЗЫВОВ\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Всего отзывов: {len(df)}\n")
        f.write(f"Позитивных: {len(df[df['sentiment'] == 1])}\n")
        f.write(f"Негативных: {len(df[df['sentiment'] == 0])}\n\n")
        f.write("ТОП-5 ТОВАРОВ ПО ДОЛЕ ПОЗИТИВА:\n")
        for i, p in enumerate(top_prods[:5], 1):  # enumerate с 1 — нумерация с единицы
            f.write(f"  {i}. {p['product']}: {p['positive_ratio']:.1%} ({p['total']} отзывов)\n")
            # :.1% — форматирование как процент с 1 знаком после запятой
        f.write("\nТОП-5 СЛОВ В НЕГАТИВНЫХ ОТЗЫВАХ:\n")
        neg_words = top_negative_words(df_binary, 5)
        for i, (word, count) in enumerate(neg_words, 1):
            f.write(f"  {i}. {word}: {count} раз\n")


# ======================================================================
# 7. STREAMLIT
# ======================================================================

def main():
    """
    Запускает веб-интерфейс Streamlit.

    Структура интерфейса:
    - Левая боковая панель (сайдбар): форма отправки отзыва
    - 4 вкладки (табы): Дашборд, Товары, Рекомендации, Модели
    - Кнопка генерации отчёта внизу
    """
    st.set_page_config(page_title="Анализатор отзывов", layout="wide")
    # set_page_config — настройка страницы (должна быть ПЕРВОЙ командой Streamlit)
    # page_icon — иконка вкладки браузера
    # layout="wide" — широкий режим (растягивается на весь экран)

    st.title("Система анализа отзывов интернет-магазина")
    st.markdown("---")  # Горизонтальная линия

    # --- Инициализация базы данных ---
    init_db()  # Создаём таблицу если её нет

    # --- Куки для сохранения user_id ---
    cookies = EncryptedCookieManager(prefix="reviews_app_", password="junior_secret_key_123")
    # EncryptedCookieManager — шифрует куки чтобы пользователь не мог их подделать
    # prefix — префикс для названий кук
    # password — пароль для шифрования

    if not cookies.ready():  # Если куки ещё не готовы
        st.stop()  # Останавливаем выполнение

    if 'user_id' not in cookies:  # Если у пользователя ещё нет ID
        cookies['user_id'] = f"user_{np.random.randint(10000, 99999)}"  # Генерируем случайный ID
        cookies.save()  # Сохраняем в куки

    user_id = cookies['user_id']  # Получаем ID текущего пользователя

    # --- Загружаем и готовим данные ---
    df = create_my_dataset()  # Создаём синтетический датасет
    df_binary = df[df['sentiment'].notna()].copy()  # Убираем записи без sentiment (нейтральные)

    # --- Обучаем модель ---
    X = df_binary['text'].values  # Тексты отзывов
    y = df_binary['sentiment'].values  # Целевая переменная (0 или 1)

    vectorizer = TfidfVectorizer(max_features=200)  # Создаём векторизатор (максимум 200 признаков)
    # max_features — ограничиваем количество самых важных слов
    X_vec = vectorizer.fit_transform(X)  # Учимся на текстах и сразу превращаем их в матрицу чисел

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    # test_size=0.2 — 20% данных на тест, 80% на обучение
    # random_state=42 — фиксируем разбиение

    results_df, model = compare_models(X_train, X_test, y_train, y_test)
    # results_df — таблица с метриками трёх моделей
    # model — LogisticRegression (выбрана как основная)

    # ================== БОКОВАЯ ПАНЕЛЬ ==================
    with st.sidebar:  # Всё внутри этого блока будет на левой панели
        st.header(f"{user_id}")  # Показываем ID пользователя
        st.markdown("---")
        st.subheader("Оставить отзыв")

        with st.form("review_form"):  # Форма (при нажатии кнопки не перезагружает всю страницу)
            product = st.selectbox("Товар:", df['product'].unique())  # Выпадающий список товаров
            rating = st.slider("Оценка (1-5):", 1, 5, 3)  # Ползунок (от 1 до 5, по умолчанию 3)
            text = st.text_area("Текст отзыва:", placeholder="Напишите ваше мнение о товаре...")
            # st.text_area — многострочное текстовое поле
            # placeholder — текст-подсказка в пустом поле

            submitted = st.form_submit_button("Отправить отзыв")  # Кнопка отправки

            if submitted and text:  # Если нажата кнопка И текст не пустой
                result, confidence = classify_text(text, model, vectorizer)
                sentiment = 1 if result == "ПОЗИТИВ" else 0
                add_review_to_db(user_id, user_id, text, rating, product, sentiment)
                st.success(f"Отзыв сохранен! Определена тональность: {result} (уверенность: {confidence:.1%})")
                # st.success — зелёное уведомление

    # ================== ВКЛАДКИ ==================
    tab1, tab2, tab3, tab4 = st.tabs(["Дашборд", "Товары", "Рекомендации", "Модели"])

    # --- Вкладка 1: Дашборд ---
    with tab1:
        st.header("Общая статистика")

        col1, col2, col3, col4 = st.columns(4)  # 4 колонки в ряд
        col1.metric("Всего отзывов", len(df))  # metric — карточка с числом
        col2.metric("Позитивных", len(df[df['sentiment'] == 1]))
        col3.metric("Негативных", len(df[df['sentiment'] == 0]))
        col4.metric("Средний рейтинг", f"{df['rating'].mean():.1f}")

        # Пытаемся показать отчёт (если он сгенерирован)
        if os.path.exists('student_report.png'):
            st.image('student_report.png', caption='Общий отчет', use_container_width=True)
            # use_container_width=True — растягиваем на ширину контейнера (может глючить)
        else:
            st.warning("Отчет еще не сгенерирован...")
            generate_empty_report()
            st.image('student_report.png', caption='Превью отчета (пустой)', use_container_width=True)

    # --- Вкладка 2: Товары ---
    with tab2:
        st.header("Статистика по товарам")

        # Собираем статистику по каждому товару
        stats_data = []
        for product in df['product'].unique():
            prod_df = df[df['product'] == product]
            pos_count = len(prod_df[prod_df['sentiment'] == 1])
            total = len(prod_df)
            stats_data.append({
                'Товар': product,
                'Количество отзывов': total,
                'Средний рейтинг': round(prod_df['rating'].mean(), 2),
                'Доля позитива': f"{pos_count / total:.1%}" if total > 0 else "0%"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)  # Интерактивная таблица

        # Облако слов ВСЕХ отзывов
        st.subheader("Облако слов всех отзывов")
        all_text = ' '.join(df['text'].values)  # Склеиваем все тексты в одну строку
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)  # Показываем график в Streamlit

    # --- Вкладка 3: Рекомендации ---
    with tab3:
        st.header("Персональные рекомендации")

        # Объединяем данные из БД и синтетического датасета
        try:
            db_reviews = get_all_reviews()
            if len(db_reviews) > 0:
                df_combined = pd.concat([df, db_reviews], ignore_index=True)
                # pd.concat — склеиваем два датафрейма
                # ignore_index=True — переназначаем индексы
            else:
                df_combined = df
        except:
            df_combined = df  # Если БД пустая или ошибка — используем только синтетику

        recs = get_recommendations_for_user(user_id, df_combined, top_n=5)

        st.subheader(f"Рекомендации для вас ({user_id}):")
        if recs:
            for i, rec in enumerate(recs, 1):
                avg_rating = df[df['product'] == rec]['rating'].mean()
                st.write(f"**{i}. {rec}** — средний рейтинг: ⭐ {avg_rating:.1f}")
        else:
            st.info("Недостаточно данных для рекомендаций. Оставьте больше отзывов!")
            # st.info — синее информационное уведомление

        # История отзывов пользователя
        st.subheader("Ваша история:")
        user_history = get_user_reviews(user_id)
        if len(user_history) > 0:
            st.dataframe(user_history[['text', 'product', 'rating', 'date']], use_container_width=True)
        else:
            st.info("У вас пока нет отзывов в базе данных.")

    # --- Вкладка 4: Модели ---
    with tab4:
        st.header("Сравнение моделей классификации")

        # Таблица с метриками (форматируем проценты)
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.2%}',  # 0.95 -> "95.00%"
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1': '{:.2%}'
        }), use_container_width=True)

        # Находим лучшую модель по F1-score
        best_model = results_df.loc[results_df['F1'].idxmax()]  # idxmax() — индекс максимального значения
        st.success(f"Лучшая модель: **{best_model['Модель']}** с F1-score = {best_model['F1']:.2%}")

        # Проверка на конкретных примерах
        st.subheader("Проверка на примерах:")
        test_phrases = [
            "Отличный товар, всем советую купить!",
            "Ужасное качество, сломался через день",
            "Нормально, но могло быть и лучше"
        ]

        for phrase in test_phrases:
            result, confidence = classify_text(phrase, model, vectorizer)
            emoji = "🟢" if result == "ПОЗИТИВ" else "🔴"
            st.write(f"{emoji} *\"{phrase}\"* → **{result}** (уверенность: {confidence:.1%})")
            # *текст* — курсив, **текст** — жирный

    # ================== КНОПКА ГЕНЕРАЦИИ ОТЧЁТА ==================
    st.markdown("---")
    if st.button("Сгенерировать student_report.png"):
        generate_report_plots(df, df_binary, model, vectorizer)
        st.success("Отчет сохранен как student_report.png и student_report.txt!")
        st.image('student_report.png', caption='Сгенерированный отчет')


# ================== ТОЧКА ВХОДА ==================
if __name__ == "__main__":
    # Это специальная переменная: равна "__main__" если файл запущен напрямую,
    # а не импортирован как модуль
    main()