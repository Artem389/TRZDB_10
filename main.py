import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import warnings
import re
import os
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib

matplotlib.use('Agg')

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager


# ======================================================================
# ФУНКЦИЯ-ЗАГЛУШКА ДЛЯ ПУСТОГО ОТЧЁТА
# ======================================================================

def generate_empty_report():
    """
    Если отчёт ещё не сгенерирован, создаём картинку-заглушку с надписью.
    """
    if not os.path.exists('student_report.png'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Отчет еще не сгенерирован\nНажмите кнопку ниже',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_title('STUDENT REPORT')
        plt.savefig('student_report.png', dpi=150, bbox_inches='tight')
        plt.close()


# ======================================================================
# 1. СОЗДАНИЕ ДАТАСЕТА (250 отзывов)
# ======================================================================

def create_my_dataset():
    """
    Генерирует синтетические отзывы.
    """
    np.random.seed(42)

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

    products = ["Смартфон X", "Ноутбук Pro", "Наушники Bass", "Часы Smart", "Планшет Lite",
                "Колонка BT", "Мышь Wireless", "Клавиатура Mech", "Монитор 4K", "Камера HD"]

    data = []

    for i in range(250):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.12, 0.2, 0.3, 0.3])
        product = np.random.choice(products)

        if rating >= 4:
            text = f"{np.random.choice(positive_words)} {product.lower()}, {np.random.choice(positive_words)}!"
            sentiment = 1
        elif rating <= 2:
            text = f"{np.random.choice(negative_words)} {product.lower()}, {np.random.choice(negative_words)}."
            sentiment = 0
        else:
            text = np.random.choice(neutral_templates) + f" {product.lower()}"
            sentiment = 0 if "ожидал" in text or "троечку" in text else 1

        data.append({
            'id': i,
            'text': text,
            'rating': rating,
            'sentiment': sentiment,
            'date': datetime.now() - timedelta(days=np.random.randint(0, 90)),
            'product': product,
            'user_name': f"Пользователь_{np.random.randint(1, 41)}",
            'user_id': np.random.randint(1, 41)
        })

    return pd.DataFrame(data)


# ======================================================================
# 2. ФУНКЦИИ ДЛЯ КЛАССИФИКАЦИИ
# ======================================================================

def classify_text(text, model, vectorizer):
    """
    Предсказывает тональность текста и возвращает уверенность модели.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = proba[int(pred)]
    result = "ПОЗИТИВ" if pred == 1 else "НЕГАТИВ"
    return result, confidence


def top_products_by_positive(df, top_n=5):
    """
    Находит топ-N товаров с самой большой долей позитивных отзывов.
    """
    result = []
    for product in df['product'].unique():
        product_data = df[df['product'] == product]
        if len(product_data) > 0:
            positive_count = len(product_data[product_data['sentiment'] == 1])
            total = len(product_data)
            ratio = positive_count / total
            result.append({'product': product, 'positive_ratio': ratio, 'total': total})

    result_sorted = sorted(result, key=lambda x: x['positive_ratio'], reverse=True)
    return result_sorted[:top_n]


def top_negative_words(df, top_n=5):
    """
    Находит самые частые слова в негативных отзывах.
    """
    negative_texts = df[df['sentiment'] == 0]['text']
    all_words = []

    for text in negative_texts:
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        all_words.extend(words)

    stop_words = ['и', 'в', 'не', 'на', 'с', 'по', 'а', 'но', 'что', 'как', 'это',
                  'из', 'у', 'я', 'за', 'от', 'для']

    filtered = [w for w in all_words if w not in stop_words and len(w) > 2]
    word_counts = Counter(filtered)
    return word_counts.most_common(top_n)


# ======================================================================
# 3. БАЗА ДАННЫХ SQLite
# ======================================================================

def init_db():
    """
    Инициализирует базу данных: создаёт таблицу reviews, если её нет.
    """
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     user_id
                     TEXT,
                     user_name
                     TEXT,
                     text
                     TEXT,
                     rating
                     INTEGER,
                     product
                     TEXT,
                     sentiment
                     INTEGER,
                     date
                     TEXT
                 )''')
    conn.commit()
    conn.close()


def add_review_to_db(user_id, user_name, text, rating, product, sentiment):
    """
    Добавляет отзыв в базу данных.
    """
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO reviews (user_id, user_name, text, rating, product, sentiment, date) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (user_id, user_name, text, rating, product, sentiment, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()


def get_all_reviews():
    """Получает ВСЕ отзывы из базы данных."""
    conn = sqlite3.connect('reviews.db')
    df = pd.read_sql_query('SELECT * FROM reviews', conn)
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


def load_all_reviews():
    """
    Загружает ВСЕ отзывы: синтетику + данные из БД.
    Если в БД есть отзывы - объединяет с синтетикой.
    Если БД пуста - возвращает только синтетику.
    """
    # Загружаем синтетические данные
    df_synthetic = create_my_dataset()

    # Пробуем загрузить данные из БД
    try:
        df_db = get_all_reviews()
        if len(df_db) > 0:
            # Объединяем синтетику и данные из БД
            df_combined = pd.concat([df_synthetic, df_db], ignore_index=True)
            return df_combined
    except:
        pass

    return df_synthetic


# ======================================================================
# 4. ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ
# ======================================================================

def get_recommendations_for_user(user_id, df_all, top_n=3):
    """
    Рекомендует товары на основе похожих пользователей.
    """
    user_reviews = df_all[df_all['user_id'] == user_id]

    if len(user_reviews) == 0:
        popular = df_all.groupby('product')['rating'].mean().sort_values(ascending=False)
        return list(popular.head(top_n).index)

    user_item = df_all.pivot_table(index='user_id', columns='product', values='rating', aggfunc='mean').fillna(0)

    try:
        user_vector = user_item.loc[user_id].values.reshape(1, -1)
    except:
        return list(df_all.groupby('product')['rating'].mean().sort_values(ascending=False).head(top_n).index)

    similarity = cosine_similarity(user_item.values, user_vector).flatten()

    user_item['similarity'] = similarity
    similar_users = user_item.sort_values('similarity', ascending=False).head(10)
    similar_users = similar_users[similar_users.index != user_id]

    products_scores = {}
    for product in df_all['product'].unique():
        if product not in user_reviews['product'].values:
            scores = []
            for sim_user in similar_users.index:
                if product in similar_users.columns:
                    val = similar_users.loc[sim_user, product]
                    sim = similar_users.loc[sim_user, 'similarity']
                    if val > 0:
                        scores.append(val * sim)
            if scores:
                products_scores[product] = np.mean(scores)

    sorted_products = sorted(products_scores.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_products[:top_n]]


# ======================================================================
# 5. СРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ
# ======================================================================

def compare_models(X_train, X_test, y_train, y_test):
    """
    Обучает и сравнивает 3 модели.
    """
    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results.append({
        'Модель': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'Recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'F1': f1_score(y_test, y_pred_lr, zero_division=0)
    })

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append({
        'Модель': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'F1': f1_score(y_test, y_pred_rf, zero_division=0)
    })

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append({
        'Модель': 'XGBoost',
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'Precision': precision_score(y_test, y_pred_xgb, zero_division=0),
        'Recall': recall_score(y_test, y_pred_xgb, zero_division=0),
        'F1': f1_score(y_test, y_pred_xgb, zero_division=0)
    })

    return pd.DataFrame(results), lr


# ======================================================================
# 6. ГЕНЕРАЦИЯ ОТЧЁТА
# ======================================================================

def generate_report_plots(df, df_binary, model, vectorizer):
    """
    Создаёт 6 графиков в одном изображении student_report.png
    и сохраняет текстовый отчёт student_report.txt
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('STUDENT REPORT - Анализ отзывов', fontsize=16, fontweight='bold')

    # [0,0] Гистограмма распределения оценок
    axes[0, 0].hist(df['rating'], bins=5, edgecolor='black', color='royalblue', alpha=0.7)
    axes[0, 0].set_title('Распределение оценок')
    axes[0, 0].set_xlabel('Оценка')
    axes[0, 0].set_ylabel('Количество')

    # [0,1] Динамика отзывов по дням
    daily = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')

    axes[0, 1].plot(daily['date'], daily['count'], 'b-o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Динамика отзывов по дням')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # [0,2] Топ товаров по доле позитива
    top_prods = top_products_by_positive(df_binary, top_n=10)
    names = [p['product'][:15] for p in top_prods]
    ratios = [p['positive_ratio'] * 100 for p in top_prods]
    colors = ['green' if r > 50 else 'red' for r in ratios]

    axes[0, 2].barh(names[::-1], ratios[::-1], color=colors[::-1])
    axes[0, 2].set_title('Доля позитивных отзывов (%)')
    axes[0, 2].set_xlim(0, 100)

    # [1,0] Матрица ошибок
    X = df_binary['text'].values
    y = df_binary['sentiment'].values
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)
    cm = confusion_matrix(y, y_pred)

    im = axes[1, 0].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

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
        wordcloud_dict = dict(negative_words)
        wc = WordCloud(width=400, height=300, background_color='white', max_words=30).generate_from_frequencies(
            wordcloud_dict)
        axes[1, 2].imshow(wc, interpolation='bilinear')
        axes[1, 2].set_title('Слова в негативных отзывах')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        axes[1, 2].set_title('Слова в негативных отзывах')

    plt.tight_layout()
    plt.savefig('student_report.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Сохраняем текстовый отчёт
    with open('student_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("ОТЧЕТ ПО АНАЛИЗУ ОТЗЫВОВ\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Всего отзывов: {len(df)}\n")
        f.write(f"Позитивных: {len(df[df['sentiment'] == 1])}\n")
        f.write(f"Негативных: {len(df[df['sentiment'] == 0])}\n\n")
        f.write("ТОП-5 ТОВАРОВ ПО ДОЛЕ ПОЗИТИВА:\n")
        for i, p in enumerate(top_prods[:5], 1):
            f.write(f"  {i}. {p['product']}: {p['positive_ratio']:.1%} ({p['total']} отзывов)\n")
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
    """
    st.set_page_config(page_title="Анализатор отзывов", layout="wide")

    st.title("Система анализа отзывов интернет-магазина")
    st.markdown("---")

    # --- Инициализация базы данных ---
    init_db()

    # --- Куки для сохранения user_id ---
    cookies = EncryptedCookieManager(prefix="reviews_app_", password="junior_secret_key_123")

    if not cookies.ready():
        st.stop()

    if 'user_id' not in cookies:
        cookies['user_id'] = f"user_{np.random.randint(10000, 99999)}"
        cookies.save()

    user_id = cookies['user_id']

    # --- Загружаем данные ДЛЯ ОБУЧЕНИЯ МОДЕЛИ (только синтетика) ---
    df_synthetic = create_my_dataset()
    df_binary = df_synthetic[df_synthetic['sentiment'].notna()].copy()

    # --- Обучаем модель на синтетике ---
    X = df_binary['text'].values
    y = df_binary['sentiment'].values

    vectorizer = TfidfVectorizer(max_features=200)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    results_df, model = compare_models(X_train, X_test, y_train, y_test)

    # --- Загружаем ВСЕ данные для отображения (синтетика + БД) ---
    df_all = load_all_reviews()

    # ================== БОКОВАЯ ПАНЕЛЬ ==================
    with st.sidebar:
        st.header(f"{user_id}")
        st.markdown("---")
        st.subheader("Оставить отзыв")

        with st.form("review_form"):
            product = st.selectbox("Товар:", df_all['product'].unique())
            rating = st.slider("Оценка (1-5):", 1, 5, 3)
            text = st.text_area("Текст отзыва:", placeholder="Напишите ваше мнение о товаре...")

            submitted = st.form_submit_button("Отправить отзыв")

            if submitted and text:
                result, confidence = classify_text(text, model, vectorizer)
                sentiment = 1 if result == "ПОЗИТИВ" else 0
                add_review_to_db(user_id, user_id, text, rating, product, sentiment)

                # Сохраняем результат в session_state перед перезагрузкой
                st.session_state[
                    'last_review_result'] = f"Отзыв сохранен! Определена тональность: {result} (уверенность: {confidence:.1%})"
                st.rerun()

    # Показываем сообщение о результате, если оно есть в session_state
    if 'last_review_result' in st.session_state and st.session_state['last_review_result']:
        st.sidebar.success(st.session_state['last_review_result'])
        # Очищаем после показа
        del st.session_state['last_review_result']

    # ================== ВКЛАДКИ ==================
    tab1, tab2, tab3, tab4 = st.tabs(["Дашборд", "Товары", "Рекомендации", "Модели"])

    # --- Вкладка 1: Дашборд ---
    with tab1:
        st.header("Общая статистика")

        # Считаем статистику по ВСЕМ данным (синтетика + БД)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего отзывов", len(df_all))
        col2.metric("Позитивных", len(df_all[df_all['sentiment'] == 1]))
        col3.metric("Негативных", len(df_all[df_all['sentiment'] == 0]))
        col4.metric("Средний рейтинг", f"{df_all['rating'].mean():.1f}")

        # Показываем отчёт
        if os.path.exists('student_report.png'):
            st.image('student_report.png', caption='Общий отчет', use_container_width=True)
        else:
            st.warning("Отчет еще не сгенерирован...")
            generate_empty_report()
            st.image('student_report.png', caption='Превью отчета (пустой)', use_container_width=True)

    # --- Вкладка 2: Товары ---
    with tab2:
        st.header("Статистика по товарам")

        stats_data = []
        for product in df_all['product'].unique():
            prod_df = df_all[df_all['product'] == product]
            pos_count = len(prod_df[prod_df['sentiment'] == 1])
            total = len(prod_df)
            stats_data.append({
                'Товар': product,
                'Количество отзывов': total,
                'Средний рейтинг': round(prod_df['rating'].mean(), 2),
                'Доля позитива': f"{pos_count / total:.1%}" if total > 0 else "0%"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Облако слов всех отзывов")
        all_text = ' '.join(df_all['text'].values)
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

    # --- Вкладка 3: Рекомендации ---
    with tab3:
        st.header("Персональные рекомендации")

        recs = get_recommendations_for_user(user_id, df_all, top_n=5)

        st.subheader(f"Рекомендации для вас ({user_id}):")
        if recs:
            for i, rec in enumerate(recs, 1):
                avg_rating = df_all[df_all['product'] == rec]['rating'].mean()
                st.write(f"**{i}. {rec}** — средний рейтинг: ⭐ {avg_rating:.1f}")
        else:
            st.info("Недостаточно данных для рекомендаций. Оставьте больше отзывов!")

        st.subheader("Ваша история:")
        user_history = get_user_reviews(user_id)
        if len(user_history) > 0:
            st.dataframe(user_history[['text', 'product', 'rating', 'date']], use_container_width=True)
        else:
            st.info("У вас пока нет отзывов в базе данных.")

    # --- Вкладка 4: Модели ---
    with tab4:
        st.header("Сравнение моделей классификации")

        st.dataframe(results_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1': '{:.2%}'
        }), use_container_width=True)

        best_model = results_df.loc[results_df['F1'].idxmax()]
        st.success(f"Лучшая модель: **{best_model['Модель']}** с F1-score = {best_model['F1']:.2%}")

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

    # ================== КНОПКА ГЕНЕРАЦИИ ОТЧЁТА ==================
    st.markdown("---")
    if st.button("Сгенерировать student_report.png"):
        # Для отчёта используем ВСЕ данные
        df_report = df_all.copy()
        df_report_binary = df_report[df_report['sentiment'].notna()].copy()
        generate_report_plots(df_report, df_report_binary, model, vectorizer)
        st.success("Отчет сохранен как student_report.png и student_report.txt!")
        st.image('student_report.png', caption='Сгенерированный отчет')


# ================== ТОЧКА ВХОДА ==================
if __name__ == "__main__":
    main()