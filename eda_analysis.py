"""Модуль исследовательского анализа и визуализации текстовых данных (EDA).

Лабораторная работа №4 по курсу «Новые технологии в РПС».
Выполнил: студент Кондратенко А.С. (гр. M_PIN_RIS_2607)
Реализует 10 аналитических задач: от токенизации до лемматизации и графиков.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import pandas as pd
import pymorphy3
import seaborn as sns

# Инициализация морфологического анализатора и стоп-слов
try:
    _STOP_WORDS = set(stopwords.words("russian"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("russian"))

_MORPH = pymorphy3.MorphAnalyzer()


# ============================================================================
# Пункт 1: Чтение данных из текстовых файлов в DataFrame
# ============================================================================
def load_reviews_to_dataframe(dataset_dir: Path | str) -> pd.DataFrame:
    """Загружает текстовые отзывы из каталога в единый DataFrame.

    :param dataset_dir: Путь к корневой папке датасета с подпапками 1..5.
    :return: DataFrame с колонками rating, title, text.
    """
    root_path = Path(dataset_dir).resolve()
    records: List[Tuple[int, str, str]] = []

    for class_folder in sorted(root_path.iterdir()):
        if not class_folder.is_dir():
            continue
        try:
            rating = int(class_folder.name)
        except ValueError:
            continue

        for file_path in sorted(class_folder.glob("*.txt")):
            try:
                raw_text = file_path.read_text(encoding="utf-8").strip()
                lines = raw_text.split("\n", 1)
                title = lines[0].strip() if lines else "Без заголовка"
                body = lines[1].strip() if len(lines) > 1 else ""
                records.append((rating, title, body))
            except OSError as err:
                print(f"[WARN] Ошибка чтения {file_path}: {err}")

    return pd.DataFrame(records, columns=["rating", "title", "text"])


# ============================================================================
# Пункт 2: Валидация и стандартизация наименований колонок
# ============================================================================
def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия столбцов к нижнему регистру и стилю snake_case.

    :param df: Исходный DataFrame.
    :return: DataFrame с переименованными колонками.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [
        re.sub(r"\W+", "_", col.strip().lower()).strip("_")
        for col in cleaned_df.columns
    ]
    return cleaned_df


# ============================================================================
# Пункт 3: Проверка и очистка невалидных значений (NaN, None, пустоты)
# ============================================================================
def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет пропуски и удаляет строки с пустыми текстами.

    :param df: Исходный DataFrame.
    :return: Очищенный от пропусков DataFrame.
    """
    cleaned = df.dropna().copy()
    # Удаляем записи, где текст состоит только из пробелов
    cleaned = cleaned[cleaned["text"].str.strip().str.len() > 0]
    return cleaned.reset_index(drop=True)


# ============================================================================
# Пункт 4: Токенизация и подсчет количества слов в отзыве
# ============================================================================
def calculate_word_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет столбец word_count с количеством слов в тексте отзыва.

    :param df: DataFrame с колонкой 'text'.
    :return: DataFrame с добавленным столбцом 'word_count'.
    """
    result = df.copy()
    # Токенизация: выделение последовательностей букв и цифр
    result["word_count"] = result["text"].apply(
        lambda t: len(re.findall(r"\b\w+\b", str(t)))
    )
    return result


# ============================================================================
# Пункт 5: Расчет описательной статистики и график boxplot
# ============================================================================
def compute_numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет числовую статистику (describe) для числовых столбцов.

    :param df: DataFrame с данными.
    :return: Таблица статистических характеристик.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return df[numeric_cols].describe()


def plot_word_count_distribution(
    df: pd.DataFrame,
    output_path: str = "word_count_boxplot.png",
) -> None:
    """Строит и сохраняет график Boxplot распределения слов по классам.

    :param df: DataFrame с колонками 'rating' и 'word_count'.
    :param output_path: Путь для сохранения изображения.
    """
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="rating",
        y="word_count",
        hue="rating", palette="Set2", legend=False,
    )
    plt.title("Распределение количества слов в отзывах по оценкам (Boxplot)")
    plt.xlabel("Оценка (Класс рейтинга)")
    plt.ylabel("Количество слов")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[ИНФО] График распределения сохранен в: {output_path}")


# ============================================================================
# Пункт 6: Фильтрация по максимальному количеству слов
# ============================================================================
def filter_by_max_word_count(
    df: pd.DataFrame,
    max_words: int,
) -> pd.DataFrame:
    """Фильтрует отзывы, оставляя тексты с длиной не более max_words.

    :param df: Исходный DataFrame.
    :param max_words: Максимальное допустимое число слов.
    :return: Отфильтрованный и отсортированный DataFrame.
    """
    filtered = df[df["word_count"] <= max_words].copy()
    return filtered.sort_values(by="word_count", ascending=False)


# ============================================================================
# Пункт 7: Фильтрация по метке класса (рейтингу)
# ============================================================================
def filter_by_star_rating(
    df: pd.DataFrame,
    rating: int | str,
) -> pd.DataFrame:
    """Фильтрует DataFrame по заданной оценке (количеству звёзд).

    :param df: Исходный DataFrame.
    :param rating: Требуемый рейтинг (1..5).
    :return: Отфильтрованный DataFrame.
    """
    target = int(rating)
    return df[df["rating"] == target].copy().reset_index(drop=True)


# ============================================================================
# Пункт 8: Группировка по классу с расчетом min, max, mean по словам
# ============================================================================
def aggregate_stats_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Группирует данные по оценке и вычисляет статистики длины отзывов.

    :param df: DataFrame с колонками 'rating' и 'word_count'.
    :return: Сводная таблица с метриками min, max, mean, median.
    """
    grouped = (
        df.groupby("rating")["word_count"]
        .agg(["min", "max", "mean", "median", "count"])
        .reset_index()
    )
    grouped.columns = [
        "rating",
        "min_words",
        "max_words",
        "mean_words",
        "median_words",
        "total_reviews",
    ]
    return grouped


# ============================================================================
# Пункт 9: Очистка текста, лемматизация и частотный анализ (NLP)
# ============================================================================
def extract_lemmatized_word_frequencies(
    df: pd.DataFrame,
    rating: int | str,
    top_n: int = 20,
) -> Dict[str, int]:
    """Выполняет токенизацию, лемматизацию и расчет частот слов для класса.

    Этапы: нижний регистр, удаление пунктуации, удаление стоп-слов,
    приведение к нормальной форме через pymorphy3.

    :param df: DataFrame с текстами.
    :param rating: Выбранный класс рейтинга.
    :param top_n: Количество наиболее частых слов.
    :return: Словарь вида {нормализованное_слово: частота}.
    """
    class_df = filter_by_star_rating(df, rating)
    tokens: List[str] = []

    # Дополнительные стоп-слова, специфичные для банковских отзывов
    custom_stops = {"это", "свой", "который", "весь", "мочь", "год"}
    all_stops = _STOP_WORDS.union(custom_stops)

    for text in class_df["text"]:
        # Удаляем пунктуацию и цифры
        cleaned = re.sub(r"[^а-яА-ЯёЁ\s]", " ", str(text).lower())
        words = cleaned.split()

        for w in words:
            if len(w) > 2 and w not in all_stops:
                # Лемматизация слова (приведение к начальной форме)
                normal_form = _MORPH.parse(w)[0].normal_form
                if normal_form not in all_stops and len(normal_form) > 2:
                    tokens.append(normal_form)

    counter = Counter(tokens)
    return dict(counter.most_common(top_n))


# ============================================================================
# Пункт 10: Построение гистограммы частотности слов
# ============================================================================
def plot_top_words_barchart(
    word_freq: Dict[str, int],
    rating: int | str,
    output_path: str = "top_words.png",
) -> None:
    """Визуализирует гистограмму наиболее частых слов для заданного класса.

    :param word_freq: Словарь частотности слов {слово: частота}.
    :param rating: Метка класса рейтинга.
    :param output_path: Путь для сохранения графика.
    """
    if not word_freq:
        print(f"[WARN] Словарь частот для класса {rating} пуст.")
        return

    words = list(word_freq.keys())
    counts = list(word_freq.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(words[::-1], counts[::-1], color="#3498db")
    plt.title(f"Топ-20 самых частых слов в отзывах с оценкой ★ {rating}")
    plt.xlabel("Количество упоминаний")
    plt.ylabel("Лемма (начальная форма слова)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(int(width)),
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[ИНФО] Гистограмма слов сохранена в: {output_path}")


# ============================================================================
# Точка входа для демонстрации всех 10 пунктов
# ============================================================================
def main() -> None:
    """Запуск последовательного выполнения всех 10 задач ЛР4."""
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print(f"[ERROR] Каталог {dataset_path} не найден!")
        return

    print("=== 1. Загрузка данных в DataFrame ===")
    df = load_reviews_to_dataframe(dataset_path)
    print(f"Всего загружено записей: {len(df)}")

    print("\n=== 2. Форматирование названий колонок ===")
    df = format_column_names(df)
    print("Колонки:", df.columns.tolist())

    print("\n=== 3. Проверка пропусков ===")
    df = clean_missing_values(df)
    print(f"Записей после очистки: {len(df)}")

    print("\n=== 4. Токенизация и подсчет слов ===")
    df = calculate_word_counts(df)
    print(df[["rating", "word_count", "title"]].head())

    print("\n=== 5. Описательная статистика длины отзывов ===")
    stats = compute_numerical_statistics(df)
    print(stats)
    plot_word_count_distribution(df, "boxplot_word_count.png")

    print("\n=== 6. Фильтрация по длине отзыва (<= 30 слов) ===")
    short_reviews = filter_by_max_word_count(df, max_words=50)
    print(f"Коротких отзывов: {len(short_reviews)}")

    print("\n=== 7. Фильтрация по классу (★ 1) ===")
    negative_reviews = filter_by_star_rating(df, rating=1)
    print(f"Отзывов с 1 звездой: {len(negative_reviews)}")

    print("\n=== 8. Группировка по классу со статистикой ===")
    grouped_stats = aggregate_stats_by_rating(df)
    print(grouped_stats)

    print("\n=== 9-10. Лемматизация и частотная гистограмма для ★ 1 ===")
    top_words_neg = extract_lemmatized_word_frequencies(df, rating=1, top_n=15)
    print("Топ слов (1 звезда):", top_words_neg)
    plot_top_words_barchart(
        top_words_neg,
        rating=1,
        output_path="top_words_1.png",
    )

    print("\n=== 9-10. Лемматизация и частотная гистограмма для ★ 5 ===")
    top_words_pos = extract_lemmatized_word_frequencies(df, rating=5, top_n=15)
    print("Топ слов (5 звезд):", top_words_pos)
    plot_top_words_barchart(
        top_words_pos,
        rating=5,
        output_path="top_words_5.png",
    )

    print("\n=== Все 10 пунктов ЛР4 успешно выполнены! ===")


if __name__ == "__main__":
    main()
