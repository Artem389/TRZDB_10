import pandas as pd  # Библиотека для работы с таблицами (DataFrame)
import numpy as np  # Библиотека для математических операций и работы с массивами
from datetime import datetime, timedelta  # Для работы с датами и временем
import sqlite3  # Для работы с базой данных SQLite
import warnings  # Для управления предупреждениями
import re  # Для работы с регулярными выражениями (поиск слов в тексте)
import os  # Для работы с файловой системой (проверка существования файлов)
from collections import Counter  # Для подсчёта частоты слов

warnings.filterwarnings('ignore')  # Отключаем вывод предупреждений (чтобы не захламлять консоль)

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

import matplotlib.pyplot as plt  # Основная библиотека для графиков
from wordcloud import WordCloud  # Красивое облако слов
import matplotlib  # Библиотека для настройки графиков

matplotlib.use('Agg')  # Говорим matplotlib работать без окна (для сервера)

import streamlit as st  # Библиотека для быстрого создания веб-приложений
from streamlit_cookies_manager import EncryptedCookieManager  # Для хранения user_id в куках


# ======================================================================
# ФУНКЦИЯ-ЗАГЛУШКА ДЛЯ ПУСТОГО ОТЧЁТА
# ======================================================================

def generate_empty_report():
    """
    Если отчёт ещё не сгенерирован, создаём картинку-заглушку с надписью.
    """
    if not os.path.exists('student_report.png'):  # Проверяем, есть ли файл (os.path.exists — проверка существования)
        fig, ax = plt.subplots(figsize=(10, 6))  # Создаём пустой холст 10x6 дюймов
        ax.text(0.5, 0.5, 'Отчет еще не сгенерирован\nНажмите кнопку ниже',  # Добавляем текст на холст
                ha='center', va='center', fontsize=16, transform=ax.transAxes)  # ha='center' — выравнивание по центру
        ax.set_title('STUDENT REPORT')  # Заголовок графика
        plt.savefig('student_report.png', dpi=150, bbox_inches='tight')  # Сохраняем в файл PNG с качеством 150 dpi
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
    positive_words = [  # Список позитивных слов
        "отличный", "прекрасный", "великолепный", "потрясающий", "качественный",
        "надежный", "стильный", "удобный", "быстрый", "красивый",
        "супер", "рекомендую", "доволен", "нравится", "лучший",
        "шикарный", "огонь", "топ", "бомба", "идеальный"
    ]

    negative_words = [  # Список негативных слов
        "ужасный", "плохой", "отвратительный", "разочарован", "кошмар",
        "брак", "сломался", "глючит", "тормозит", "деньги на ветер",
        "не советую", "не покупайте", "фигня", "ерунда", "отстой",
        "хлам", "мусор", "никакой", "бесполезный", "ужас"
    ]

    neutral_templates = [  # Нейтральные шаблоны (для 3 звёзд)
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

    products = ["Смартфон X", "Ноутбук Pro", "Наушники Bass", "Часы Smart", "Планшет Lite",  # Список товаров
                "Колонка BT", "Мышь Wireless", "Клавиатура Mech", "Монитор 4K", "Камера HD"]

    data = []  # Пустой список для будущих записей

    for i in range(250):  # Генерируем 250 записей
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.12, 0.2, 0.3, 0.3])  # Выбираем рейтинг с вероятностями
        # 8% шанс на 1 звезду, 12% на 2 звезды, 20% на 3, 30% на 4, 30% на 5

        product = np.random.choice(products)  # Случайный товар

        # ЛОГИКА ГЕНЕРАЦИИ ТЕКСТА В ЗАВИСИМОСТИ ОТ РЕЙТИНГА
        if rating >= 4:  # Если оценка 4-5
            text = f"{np.random.choice(positive_words)} {product.lower()}, {np.random.choice(positive_words)}!"  # Позитивный текст
            sentiment = 1  # 1 = позитив
        elif rating <= 2:  # Если оценка 1-2
            text = f"{np.random.choice(negative_words)} {product.lower()}, {np.random.choice(negative_words)}."  # Негативный текст
            sentiment = 0  # 0 = негатив
        else:  # Оценка 3
            text = np.random.choice(neutral_templates) + f" {product.lower()}"  # Нейтральный текст
            sentiment = 0 if "ожидал" in text or "троечку" in text else 1  # Определяем тональность по тексту

        data.append({  # Добавляем запись в список
            'id': i,  # ID отзыва
            'text': text,  # Текст отзыва
            'rating': rating,  # Оценка пользователя (1-5)
            'sentiment': sentiment,  # Тональность (0=негатив, 1=позитив)
            'date': datetime.now() - timedelta(days=np.random.randint(0, 90)),
            # Дата отзыва (случайная в пределах 90 дней)
            'product': product,  # Название товара
            'user_name': f"Пользователь_{np.random.randint(1, 41)}",  # Имена от 1 до 40
            'user_id': np.random.randint(1, 41)  # ID от 1 до 40
        })

    return pd.DataFrame(data)  # Превращаем список словарей в таблицу DataFrame


# ======================================================================
# 2. ФУНКЦИИ ДЛЯ КЛАССИФИКАЦИИ
# ======================================================================

def classify_text(text, model, vectorizer):
    """
    Предсказывает тональность текста и возвращает уверенность модели.
    """
    X = vectorizer.transform([text])  # Превращаем текст в массив чисел (TF-IDF)
    pred = model.predict(X)[0]  # Предсказываем класс (0 или 1), берём первый результат
    proba = model.predict_proba(X)[0]  # Получаем вероятности для всех классов
    confidence = proba[int(pred)]  # Берём вероятность предсказанного класса
    result = "ПОЗИТИВ" if pred == 1 else "НЕГАТИВ"  # Тернарный оператор (сокращённый if)
    return result, confidence  # Возвращаем результат и уверенность


def top_products_by_positive(df, top_n=5):
    """
    Находит топ-N товаров с самой большой долей позитивных отзывов.
    """
    result = []  # Пустой список для результатов
    for product in df['product'].unique():  # Проходим по уникальным товарам
        product_data = df[df['product'] == product]  # Фильтруем только отзывы на этот товар
        if len(product_data) > 0:  # Если есть хоть один отзыв
            positive_count = len(product_data[product_data['sentiment'] == 1])  # Считаем позитивные
            total = len(product_data)  # Всего отзывов
            ratio = positive_count / total  # Доля позитивных (от 0 до 1)
            result.append({'product': product, 'positive_ratio': ratio, 'total': total})  # Добавляем в список

    result_sorted = sorted(result, key=lambda x: x['positive_ratio'], reverse=True)  # Сортируем по убыванию
    return result_sorted[:top_n]  # Возвращаем только первые top_n


def top_negative_words(df, top_n=5):
    """
    Находит самые частые слова в негативных отзывах.
    """
    negative_texts = df[df['sentiment'] == 0]['text']  # Берём только негативные отзывы
    all_words = []  # Список для всех слов

    for text in negative_texts:  # Проходим по каждому отзыву
        words = re.findall(r'\b[а-яё]+\b', text.lower())  # Ищем все русские слова (регулярное выражение)
        all_words.extend(words)  # Добавляем найденные слова в общий список

    stop_words = ['и', 'в', 'не', 'на', 'с', 'по', 'а', 'но', 'что', 'как', 'это',  # Стоп-слова
                  'из', 'у', 'я', 'за', 'от', 'для']

    filtered = [w for w in all_words if w not in stop_words and len(w) > 2]  # Убираем стоп-слова и короткие слова
    word_counts = Counter(filtered)  # Считаем частоту каждого слова
    return word_counts.most_common(top_n)  # Возвращаем самые частые элементы


# ======================================================================
# 3. БАЗА ДАННЫХ SQLite
# ======================================================================

def init_db():
    """
    Инициализирует базу данных: создаёт таблицу reviews, если её нет.
    """
    conn = sqlite3.connect('reviews.db')  # Подключаемся к файлу reviews.db (создаётся если нет)
    c = conn.cursor()  # Курсор — объект для выполнения SQL-запросов
    c.execute('''CREATE TABLE IF NOT EXISTS reviews # Создаём таблицу если её нет
    (
        id
        INTEGER
        PRIMARY
        KEY
        AUTOINCREMENT,
        #
        Автоинкрементный
        ID
        user_id
        TEXT,
        #
        ID
        пользователя
                 (
        текстовый
                 )
        user_name TEXT, # Имя пользователя
        text TEXT, # Текст отзыва
        rating INTEGER, # Оценка
                 (
                     1
                     -
                     5
                 )
        product TEXT, # Название товара
        sentiment INTEGER, # Тональность
                 (
                     0=
                     негатив,
                     1=
                     позитив
                 )
        date TEXT # Дата отзыва
        )''')
    conn.commit()  # Сохраняем изменения
    conn.close()  # Закрываем соединение


def add_review_to_db(user_id, user_name, text, rating, product, sentiment):
    """
    Добавляет отзыв в базу данных.
    """
    conn = sqlite3.connect('reviews.db')  # Подключаемся к БД
    c = conn.cursor()  # Создаём курсор
    c.execute(  # Выполняем SQL-запрос с плейсхолдерами
        'INSERT INTO reviews (user_id, user_name, text, rating, product, sentiment, date) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, user_name, text, rating, product, sentiment,
         datetime.now().strftime('%Y-%m-%d %H:%M:%S')))  # Подставляем значения
    conn.commit()  # Сохраняем изменения
    conn.close()  # Закрываем соединение


def get_all_reviews():
    """Получает ВСЕ отзывы из базы данных."""
    conn = sqlite3.connect('reviews.db')  # Подключаемся к БД
    df = pd.read_sql_query('SELECT * FROM reviews', conn)  # Выполняем запрос и сразу получаем DataFrame
    conn.close()  # Закрываем соединение
    return df  # Возвращаем таблицу


def get_user_reviews(user_id):
    """
    Получает отзывы конкретного пользователя.
    """
    conn = sqlite3.connect('reviews.db')  # Подключаемся к БД
    df = pd.read_sql_query(f"SELECT * FROM reviews WHERE user_id='{user_id}'", conn)  # Запрос с фильтром по user_id
    conn.close()  # Закрываем соединение
    return df  # Возвращаем таблицу


def load_all_reviews():
    """
    Загружает ВСЕ отзывы: синтетику + данные из БД.
    """
    df_synthetic = create_my_dataset()  # Создаём синтетические данные

    try:  # Пробуем загрузить данные из БД
        df_db = get_all_reviews()  # Получаем отзывы из БД
        if len(df_db) > 0:  # Если в БД есть отзывы
            df_combined = pd.concat([df_synthetic, df_db], ignore_index=True)  # Объединяем синтетику и БД
            return df_combined  # Возвращаем объединённые данные
    except:  # Если ошибка (например, таблица пуста)
        pass  # Пропускаем

    return df_synthetic  # Возвращаем только синтетику


# ======================================================================
# 4. ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ
# ======================================================================

def get_recommendations_for_user(user_id, df_all, top_n=3):
    """
    Рекомендует товары на основе похожих пользователей.
    """
    user_reviews = df_all[df_all['user_id'] == user_id]  # Отзывы текущего пользователя

    if len(user_reviews) == 0:  # Если пользователь новый (нет отзывов)
        popular = df_all.groupby('product')['rating'].mean().sort_values(ascending=False)  # Самые популярные товары
        return list(popular.head(top_n).index)  # Возвращаем названия товаров

    user_item = df_all.pivot_table(index='user_id', columns='product', values='rating', aggfunc='mean').fillna(0)
    # Строим матрицу: строки = пользователи, столбцы = товары, значения = средний рейтинг

    try:  # Пытаемся получить вектор текущего пользователя
        user_vector = user_item.loc[user_id].values.reshape(1, -1)  # Берём строку пользователя и превращаем в массив
    except:  # Если пользователя нет в матрице
        return list(df_all.groupby('product')['rating'].mean().sort_values(ascending=False).head(
            top_n).index)  # Возвращаем популярное

    similarity = cosine_similarity(user_item.values, user_vector).flatten()  # Считаем косинусное сходство

    user_item['similarity'] = similarity  # Добавляем столбец со сходством
    similar_users = user_item.sort_values('similarity', ascending=False).head(10)  # Берём топ-10 похожих
    similar_users = similar_users[similar_users.index != user_id]  # Убираем самого пользователя

    products_scores = {}  # Словарь для оценок товаров
    for product in df_all['product'].unique():  # Проходим по всем товарам
        if product not in user_reviews['product'].values:  # Только товары, которые пользователь не оценивал
            scores = []  # Список оценок от похожих пользователей
            for sim_user in similar_users.index:  # Проходим по похожим пользователям
                if product in similar_users.columns:  # Если товар есть в матрице
                    val = similar_users.loc[sim_user, product]  # Оценка похожего пользователя
                    sim = similar_users.loc[sim_user, 'similarity']  # Сходство с ним
                    if val > 0:  # Если оценка положительная
                        scores.append(val * sim)  # Взвешенная оценка (оценка * сходство)
            if scores:  # Если есть оценки
                products_scores[product] = np.mean(scores)  # Средняя взвешенная оценка

    sorted_products = sorted(products_scores.items(), key=lambda x: x[1], reverse=True)  # Сортируем по рейтингу
    return [p[0] for p in sorted_products[:top_n]]  # Возвращаем только названия товаров


# ======================================================================
# 5. СРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ
# ======================================================================

def compare_models(X_train, X_test, y_train, y_test):
    """
    Обучает и сравнивает 3 модели.
    """
    results = []  # Список для результатов

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)  # Создаём модель с максимальным числом итераций
    lr.fit(X_train, y_train)  # Обучаем модель
    y_pred_lr = lr.predict(X_test)  # Предсказываем на тестовых данных
    results.append({  # Добавляем результаты
        'Модель': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_lr),  # Доля правильных ответов
        'Precision': precision_score(y_test, y_pred_lr, zero_division=0),  # Точность
        'Recall': recall_score(y_test, y_pred_lr, zero_division=0),  # Полнота
        'F1': f1_score(y_test, y_pred_lr, zero_division=0)  # F1-мера
    })

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 деревьев, фиксируем сид
    rf.fit(X_train, y_train)  # Обучаем
    y_pred_rf = rf.predict(X_test)  # Предсказываем
    results.append({  # Добавляем результаты
        'Модель': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'F1': f1_score(y_test, y_pred_rf, zero_division=0)
    })

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  # Бустинг
    xgb.fit(X_train, y_train)  # Обучаем
    y_pred_xgb = xgb.predict(X_test)  # Предсказываем
    results.append({  # Добавляем результаты
        'Модель': 'XGBoost',
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'Precision': precision_score(y_test, y_pred_xgb, zero_division=0),
        'Recall': recall_score(y_test, y_pred_xgb, zero_division=0),
        'F1': f1_score(y_test, y_pred_xgb, zero_division=0)
    })

    return pd.DataFrame(results), lr  # Возвращаем таблицу с метриками и LR как основную модель


# ======================================================================
# 6. ГЕНЕРАЦИЯ ОТЧЁТА
# ======================================================================

def generate_report_plots(df, df_binary, model, vectorizer):
    """
    Создаёт 6 графиков в одном изображении student_report.png
    и сохраняет текстовый отчёт student_report.txt
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Создаём холст 2x3 графика
    fig.suptitle('STUDENT REPORT - Анализ отзывов', fontsize=16, fontweight='bold')  # Общий заголовок

    # [0,0] Гистограмма распределения оценок
    axes[0, 0].hist(df['rating'], bins=5, edgecolor='black', color='royalblue', alpha=0.7)  # Гистограмма
    axes[0, 0].set_title('Распределение оценок')  # Заголовок
    axes[0, 0].set_xlabel('Оценка')  # Подпись оси X
    axes[0, 0].set_ylabel('Количество')  # Подпись оси Y

    # [0,1] Динамика отзывов по дням
    daily = df.groupby(df['date'].dt.date).size().reset_index(name='count')  # Группируем по датам
    daily['date'] = pd.to_datetime(daily['date'])  # Превращаем обратно в datetime
    daily = daily.sort_values('date')  # Сортируем по дате

    axes[0, 1].plot(daily['date'], daily['count'], 'b-o', linewidth=2, markersize=4)  # Линейный график
    axes[0, 1].set_title('Динамика отзывов по дням')  # Заголовок
    axes[0, 1].tick_params(axis='x', rotation=45)  # Поворачиваем подписи по оси X

    # [0,2] Топ товаров по доле позитива
    top_prods = top_products_by_positive(df_binary, top_n=10)  # Получаем топ-10 товаров
    names = [p['product'][:15] for p in top_prods]  # Обрезаем названия до 15 символов
    ratios = [p['positive_ratio'] * 100 for p in top_prods]  # Переводим в проценты
    colors = ['green' if r > 50 else 'red' for r in ratios]  # Зелёный если >50%, красный если <50%

    axes[0, 2].barh(names[::-1], ratios[::-1], color=colors[::-1])  # Горизонтальная гистограмма
    axes[0, 2].set_title('Доля позитивных отзывов (%)')  # Заголовок
    axes[0, 2].set_xlim(0, 100)  # Ось X от 0 до 100%

    # [1,0] Матрица ошибок
    X = df_binary['text'].values  # Берём тексты
    y = df_binary['sentiment'].values  # Берём метки
    X_vec = vectorizer.transform(X)  # Векторизуем
    y_pred = model.predict(X_vec)  # Предсказываем
    cm = confusion_matrix(y, y_pred)  # Строим матрицу ошибок

    im = axes[1, 0].imshow(cm, cmap='Blues')  # Показываем матрицу цветом
    for i in range(2):  # Для каждой строки
        for j in range(2):  # Для каждого столбца
            axes[1, 0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)  # Пишем число в ячейке

    axes[1, 0].set_title('Матрица ошибок')  # Заголовок
    axes[1, 0].set_xticks([0, 1])  # Метки по оси X
    axes[1, 0].set_yticks([0, 1])  # Метки по оси Y
    axes[1, 0].set_xticklabels(['Негатив', 'Позитив'])  # Подписи
    axes[1, 0].set_yticklabels(['Негатив', 'Позитив'])  # Подписи

    # [1,1] Средний рейтинг товаров
    avg_rating = df.groupby('product')['rating'].mean().sort_values()  # Средний рейтинг по товарам
    axes[1, 1].barh(avg_rating.index, avg_rating.values, color='seagreen', alpha=0.7)  # Горизонтальная гистограмма
    axes[1, 1].set_title('Средний рейтинг товаров')  # Заголовок
    axes[1, 1].set_xlim(0, 5)  # Ось X от 0 до 5

    # [1,2] Облако слов негативных отзывов
    negative_words = top_negative_words(df_binary, top_n=30)  # Получаем топ-30 негативных слов
    if negative_words:  # Если есть слова
        wordcloud_dict = dict(negative_words)  # Превращаем в словарь
        wc = WordCloud(width=400, height=300, background_color='white', max_words=30).generate_from_frequencies(
            wordcloud_dict)
        # Создаём облако слов
        axes[1, 2].imshow(wc, interpolation='bilinear')  # Показываем облако
        axes[1, 2].set_title('Слова в негативных отзывах')  # Заголовок
        axes[1, 2].axis('off')  # Убираем оси
    else:  # Если слов нет
        axes[1, 2].text(0.5, 0.5, 'Нет данных', ha='center', va='center')  # Пишем "Нет данных"
        axes[1, 2].set_title('Слова в негативных отзывах')  # Заголовок

    plt.tight_layout()  # Автоматически подгоняем чтобы не накладывались
    plt.savefig('student_report.png', dpi=150, bbox_inches='tight')  # Сохраняем
    plt.close()  # Закрываем

    # --- СОХРАНЯЕМ ТЕКСТОВЫЙ ОТЧЁТ ---
    with open('student_report.txt', 'w', encoding='utf-8') as f:  # Открываем файл для записи
        f.write("=" * 50 + "\n")  # Разделитель
        f.write("ОТЧЕТ ПО АНАЛИЗУ ОТЗЫВОВ\n")  # Заголовок
        f.write("=" * 50 + "\n\n")  # Разделитель
        f.write(f"Всего отзывов: {len(df)}\n")  # Общее количество
        f.write(f"Позитивных: {len(df[df['sentiment'] == 1])}\n")  # Позитивные
        f.write(f"Негативных: {len(df[df['sentiment'] == 0])}\n\n")  # Негативные
        f.write("ТОП-5 ТОВАРОВ ПО ДОЛЕ ПОЗИТИВА:\n")  # Заголовок
        for i, p in enumerate(top_prods[:5], 1):  # Проходим по топ-5
            f.write(f"  {i}. {p['product']}: {p['positive_ratio']:.1%} ({p['total']} отзывов)\n")  # Записываем
        f.write("\nТОП-5 СЛОВ В НЕГАТИВНЫХ ОТЗЫВАХ:\n")  # Заголовок
        neg_words = top_negative_words(df_binary, 5)  # Получаем топ-5 негативных слов
        for i, (word, count) in enumerate(neg_words, 1):  # Проходим по топ-5
            f.write(f"  {i}. {word}: {count} раз\n")  # Записываем


# ======================================================================
# 7. STREAMLIT
# ======================================================================

def main():
    """
    Запускает веб-интерфейс Streamlit.
    """
    st.set_page_config(page_title="Анализатор отзывов", layout="wide")  # Настройка страницы

    st.title("Система анализа отзывов интернет-магазина")  # Заголовок
    st.markdown("---")  # Горизонтальная линия

    # --- Инициализация базы данных ---
    init_db()  # Создаём таблицу если её нет

    # --- Куки для сохранения user_id ---
    cookies = EncryptedCookieManager(prefix="reviews_app_", password="junior_secret_key_123")  # Шифрованные куки

    if not cookies.ready():  # Если куки ещё не готовы
        st.stop()  # Останавливаем выполнение

    if 'user_id' not in cookies:  # Если у пользователя ещё нет ID
        cookies['user_id'] = f"user_{np.random.randint(10000, 99999)}"  # Генерируем случайный ID
        cookies.save()  # Сохраняем в куки

    user_id = cookies['user_id']  # Получаем ID текущего пользователя

    # --- Загружаем данные ДЛЯ ОБУЧЕНИЯ МОДЕЛИ (только синтетика) ---
    df_synthetic = create_my_dataset()  # Создаём синтетический датасет
    df_binary = df_synthetic[df_synthetic['sentiment'].notna()].copy()  # Убираем записи без sentiment

    # --- Обучаем модель на синтетике ---
    X = df_binary['text'].values  # Тексты отзывов
    y = df_binary['sentiment'].values  # Целевая переменная

    vectorizer = TfidfVectorizer(max_features=200)  # Создаём векторизатор (максимум 200 признаков)
    X_vec = vectorizer.fit_transform(X)  # Учимся на текстах и превращаем в числа

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2,
                                                        random_state=42)  # Делим на обучение и тест

    results_df, model = compare_models(X_train, X_test, y_train, y_test)  # Сравниваем модели

    # --- Загружаем ВСЕ данные для отображения (синтетика + БД) ---
    df_all = load_all_reviews()  # Загружаем объединённые данные

    # ================== БОКОВАЯ ПАНЕЛЬ ==================
    with st.sidebar:  # Всё внутри этого блока будет на левой панели
        st.header(f"{user_id}")  # Показываем ID пользователя
        st.markdown("---")  # Разделитель
        st.subheader("Оставить отзыв")  # Заголовок

        with st.form("review_form"):  # Форма (при нажатии не перезагружает всю страницу)
            product = st.selectbox("Товар:", df_all['product'].unique())  # Выпадающий список товаров
            rating = st.slider("Оценка (1-5):", 1, 5, 3)  # Ползунок
            text = st.text_area("Текст отзыва:", placeholder="Напишите ваше мнение о товаре...")  # Текстовое поле

            submitted = st.form_submit_button("Отправить отзыв")  # Кнопка отправки

            if submitted and text:  # Если нажата кнопка И текст не пустой
                result, confidence = classify_text(text, model, vectorizer)  # Определяем тональность
                sentiment = 1 if result == "ПОЗИТИВ" else 0  # Превращаем в 0 или 1
                add_review_to_db(user_id, user_id, text, rating, product, sentiment)  # Сохраняем в БД

                st.session_state[
                    'last_review_result'] = f"Отзыв сохранен! Определена тональность: {result} (уверенность: {confidence:.1%})"  # Сохраняем в сессию
                st.rerun()  # Обновляем страницу

    # Показываем сообщение о результате, если оно есть
    if 'last_review_result' in st.session_state and st.session_state['last_review_result']:
        st.sidebar.success(st.session_state['last_review_result'])  # Показываем зелёное уведомление
        del st.session_state['last_review_result']  # Очищаем после показа

    # ================== ВКЛАДКИ ==================
    tab1, tab2, tab3, tab4 = st.tabs(["Дашборд", "Товары", "Рекомендации", "Модели"])  # Создаём вкладки

    # --- Вкладка 1: Дашборд ---
    with tab1:
        st.header("Общая статистика")  # Заголовок

        col1, col2, col3, col4 = st.columns(4)  # 4 колонки в ряд
        col1.metric("Всего отзывов", len(df_all))  # Карточка с числом
        col2.metric("Позитивных", len(df_all[df_all['sentiment'] == 1]))  # Карточка
        col3.metric("Негативных", len(df_all[df_all['sentiment'] == 0]))  # Карточка
        col4.metric("Средний рейтинг", f"{df_all['rating'].mean():.1f}")  # Карточка

        if os.path.exists('student_report.png'):  # Если отчёт существует
            st.image('student_report.png', caption='Общий отчет', use_container_width=True)  # Показываем
        else:  # Если отчёта нет
            st.warning("Отчет еще не сгенерирован...")  # Предупреждение
            generate_empty_report()  # Создаём заглушку
            st.image('student_report.png', caption='Превью отчета (пустой)', use_container_width=True)  # Показываем

    # --- Вкладка 2: Товары ---
    with tab2:
        st.header("Статистика по товарам")  # Заголовок

        stats_data = []  # Список для статистики
        for product in df_all['product'].unique():  # Проходим по товарам
            prod_df = df_all[df_all['product'] == product]  # Отзывы на товар
            pos_count = len(prod_df[prod_df['sentiment'] == 1])  # Позитивные
            total = len(prod_df)  # Всего
            stats_data.append({  # Добавляем в список
                'Товар': product,
                'Количество отзывов': total,
                'Средний рейтинг': round(prod_df['rating'].mean(), 2),
                'Доля позитива': f"{pos_count / total:.1%}" if total > 0 else "0%"
            })

        stats_df = pd.DataFrame(stats_data)  # Превращаем в таблицу
        st.dataframe(stats_df, use_container_width=True)  # Показываем таблицу

        st.subheader("Облако слов всех отзывов")  # Заголовок
        all_text = ' '.join(df_all['text'].values)  # Склеиваем все тексты
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(
            all_text)  # Создаём облако
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))  # Создаём холст
        ax_wc.imshow(wc, interpolation='bilinear')  # Показываем облако
        ax_wc.axis('off')  # Убираем оси
        st.pyplot(fig_wc)  # Показываем график

    # --- Вкладка 3: Рекомендации ---
    with tab3:
        st.header("Персональные рекомендации")  # Заголовок

        recs = get_recommendations_for_user(user_id, df_all, top_n=5)  # Получаем рекомендации

        st.subheader(f"Рекомендации для вас ({user_id}):")  # Заголовок
        if recs:  # Если есть рекомендации
            for i, rec in enumerate(recs, 1):  # Проходим по ним
                avg_rating = df_all[df_all['product'] == rec]['rating'].mean()  # Средний рейтинг товара
                st.write(f"**{i}. {rec}** — средний рейтинг: ⭐ {avg_rating:.1f}")  # Показываем
        else:  # Если рекомендаций нет
            st.info("Недостаточно данных для рекомендаций. Оставьте больше отзывов!")  # Информационное сообщение

        st.subheader("Ваша история:")  # Заголовок
        user_history = get_user_reviews(user_id)  # Получаем отзывы пользователя
        if len(user_history) > 0:  # Если есть отзывы
            st.dataframe(user_history[['text', 'product', 'rating', 'date']], use_container_width=True)  # Показываем
        else:  # Если отзывов нет
            st.info("У вас пока нет отзывов в базе данных.")  # Информационное сообщение

    # --- Вкладка 4: Модели ---
    with tab4:
        st.header("Сравнение моделей классификации")  # Заголовок

        st.dataframe(results_df.style.format({  # Показываем таблицу с форматированием
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1': '{:.2%}'
        }), use_container_width=True)

        best_model = results_df.loc[results_df['F1'].idxmax()]  # Находим лучшую модель
        st.success(f"Лучшая модель: **{best_model['Модель']}** с F1-score = {best_model['F1']:.2%}")  # Показываем

        st.subheader("Проверка на примерах:")  # Заголовок
        test_phrases = [  # Тестовые фразы
            "Отличный товар, всем советую купить!",
            "Ужасное качество, сломался через день",
            "Нормально, но могло быть и лучше"
        ]

        for phrase in test_phrases:  # Проходим по фразам
            result, confidence = classify_text(phrase, model, vectorizer)  # Определяем тональность
            emoji = "🟢" if result == "ПОЗИТИВ" else "🔴"  # Выбираем эмодзи
            st.write(f"{emoji} *\"{phrase}\"* → **{result}** (уверенность: {confidence:.1%})")  # Показываем

    # ================== КНОПКА ГЕНЕРАЦИИ ОТЧЁТА ==================
    st.markdown("---")  # Разделитель
    if st.button("Сгенерировать student_report.png"):  # Кнопка
        df_report = df_all.copy()  # Копируем данные
        df_report_binary = df_report[df_report['sentiment'].notna()].copy()  # Убираем без sentiment
        generate_report_plots(df_report, df_report_binary, model, vectorizer)  # Генерируем отчёт
        st.success("Отчет сохранен как student_report.png и student_report.txt!")  # Успех
        st.image('student_report.png', caption='Сгенерированный отчет')  # Показываем


# ================== ТОЧКА ВХОДА ==================
if __name__ == "__main__":  # Проверяем, запущен ли файл напрямую
    main()  # Запускаем основную функцию