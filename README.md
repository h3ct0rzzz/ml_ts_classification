# Machine Learning project | TimeSeries binary classification.

---
## Организация проекта
```plaintext
    ├── LICENSE
    ├── Makefile           <- Makefile с командами типа `make data` или `make train`
    ├── README.md          <- README файл.
    ├── data
    │   ├── processed      <- Окончательные наборы данных для моделирования.
    │   └── raw            <- Исходный, неизменяемый дамп данных.
    │
    ├── models             <- Обученные и сериализованные модели.
    │
    ├── notebooks          <- Jupyter блокноты.
    │
    ├── requirements.txt   <- Файл зависимостей для воспроизведения среды анализа,
    │                         созданный с помощью `pip freeze > requirements.txt`
    │
    ├── setup.py           <- делает проект устанавливаемым через pip (pip install -e .) 
    │                         для импорта src
    ├── src                <- Исходный код для использования в этом проекте.
    │   ├── __init__.py    <- Делает src Python модулем
    │   │
    │   ├── data           <- Скрипты для загрузки или генерации данных
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Скрипты для Feature Engineering'a.
    │   │   └── build_features.py
    │   │
    │   ├──models          <- Скрипты для обучения моделей и использования обученных моделей
    │   │   │                 для предсказаний
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── constants.py   <- Константы для проекта.
    │
    └── tox.ini            <- tox файл с настройками для запуска tox; 
```
---

## Документация по скриптам.

### `src/features/build_features.py` - скрипт для feature engineering'a.

Каждая функция декорирована с помощью `@feature_function`.

#### Функции:

##### 1. `linear_trend(data)`
- **Описание**: Вычисляет наклон линейного тренда данных.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

##### 2. `phase_duration(data)`
- **Описание**: Определяет максимальную длительность фазы выше/ниже среднего значения.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int`

##### 3. `max_trend_length(data)`
- **Описание**: Находит длину самой длинной монотонной последовательности.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int`

##### 4. `time_series_entropy(data)`
- **Описание**: Вычисляет энтропию распределения данных.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 5. `calculate_iqr(data)`
- **Описание**: Вычисляет интерквартильный размах (IQR).
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

##### 6. `variance_larger_than_standard_deviation(data)`
- **Описание**: Проверяет превышение дисперсии над стандартным отклонением.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int` (0 или 1)

##### 7. `has_duplicate_max(data)`
- **Описание**: Проверяет повторение максимального значения.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int` (0 или 1)

#### 8. `has_duplicate_min(data)`
- **Описание**: Проверяет повторение минимального значения.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int` (0 или 1)

#### 9. `has_duplicates(data)`
- **Описание**: Проверяет наличие дубликатов.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int` (0 или 1)

#### 10. `sum_values(data)`
- **Описание**: Вычисляет сумму значений.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 11. `abs_energy(data)`
- **Описание**: Вычисляет абсолютную энергию.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 12. `complexity_invariant_distance(data, normalize=False)`
- **Описание**: Вычисляет инвариантное расстояние сложности.
- **input**: `List[float]` или `np.ndarray`, `bool`
- **output**: `float`

#### 13. `mean_abs_change(data)`
- **Описание**: Вычисляет среднее абсолютное изменение.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 14. `mean_change(data)`
- **Описание**: Вычисляет среднее изменение.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 15. `mean_second_derivative_central(data)`
- **Описание**: Вычисляет среднюю вторую производную.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 16. `median(data)`
- **Описание**: Вычисляет медиану.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 17. `mean(data)`
- **Описание**: Вычисляет среднее значение.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 18. `length(data)`
- **Описание**: Возвращает длину временного ряда.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int`

#### 19. `standard_deviation(data)`
- **Описание**: Вычисляет стандартное отклонение.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 20. `variation_coefficient(data)`
- **Описание**: Вычисляет коэффициент вариации.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 21. `variance(data)`
- **Описание**: Вычисляет дисперсию.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 22. `skewness(data)`
- **Описание**: Вычисляет асимметрию.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 23. `kurtosis(data)`
- **Описание**: Вычисляет эксцесс.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 24. `root_mean_square(data)`
- **Описание**: Вычисляет среднеквадратичное значение.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 25. `absolute_sum_of_changes(data)`
- **Описание**: Вычисляет сумму абсолютных изменений.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 26. `count_above_mean(data)`
- **Описание**: Считает значения выше среднего.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int`

#### 27. `count_below_mean(data)`
- **Описание**: Считает значения ниже среднего.
- **input**: `List[float]` или `np.ndarray`
- **output**: `int`

#### 28. `last_location_of_maximum(data)`
- **Описание**: Находит последнюю позицию максимума.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 29. `first_location_of_maximum(data)`
- **Описание**: Находит первую позицию максимума.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 30. `last_location_of_minimum(data)`
- **Описание**: Находит последнюю позицию минимума.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 31. `first_location_of_minimum(data)`
- **Описание**: Находит первую позицию минимума.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 32. `percentage_of_reoccurring_values(data)`
- **Описание**: Вычисляет процент повторяющихся значений.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 33. `percentage_of_reoccurring_datapoints(data)`
- **Описание**: Вычисляет процент повторяющихся точек.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 34. `sum_of_reoccurring_values(data)`
- **Описание**: Суммирует повторяющиеся значения.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 35. `ratio_unique_values(data)`
- **Описание**: Вычисляет долю уникальных значений.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 36. `quantile(data, quantile_value)`
- **Описание**: Вычисляет квантиль.
- **input**: `List[float]` или `np.ndarray`, `float`
- **output**: `float`

#### 37. `maximum(data)`
- **Описание**: Находит максимум.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 38. `absolute_maximum(data)`
- **Описание**: Находит абсолютный максимум.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 39. `minimum(data)`
- **Описание**: Находит минимум.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 40. `benford_correlation(data)`
- **Описание**: Вычисляет корреляцию с законом Бенфорда.
- **input**: `List[float]` или `np.ndarray`
- **output**: `float`

#### 41. `autocorrelation(data, lag)`
- **Описание**: Вычисляет автокорреляцию.
- **input**: `List[float]` или `np.ndarray`, `int`
- **output**: `float`

#### 42. `fft_coefficient(data, coefficient=0)`
- **Описание**: Вычисляет коэффициент БПФ.
- **input**: `List[float]` или `np.ndarray`, `int`
- **output**: `complex`

#### 43. `number_crossing_m(data, threshold)`
- **Описание**: Считает пересечения порога.
- **input**: `List[float]` или `np.ndarray`, `float`
- **output**: `int`

#### 44. `energy_ratio_by_chunks(data, num_segments, segment_focus)`
- **Описание**: Вычисляет отношение энергии сегментов.
- **input**: `List[float]` или `np.ndarray`, `int`, `int`
- **output**: `float`

#### 45. `permutation_entropy(data, order, delay)`
- **Описание**: Вычисляет энтропию перестановок.
- **input**: `List[float]` или `np.ndarray`, `int`, `int`
- **output**: `float`

---

### `src/data/make_dataset.py` - основные функции для сборки датасета.

#### Функции:

##### 1. `fill_missing_values(df: pd.DataFrame) -> pd.DataFrame`
- **Описание**: Заполняет пропущенные значения в DataFrame, случайно выбирая из доступных непустых значений в каждом столбце.
- **input**: `pd.DataFrame`
- **output**: `pd.DataFrame`

##### 2. `generate_features(df: pd.DataFrame) -> pd.DataFrame`
- **Описание**: Генерирует признаки из DataFrame, применяя различные функции генерации признаков к столбцу 'values'.
- **input**: `pd.DataFrame`
- **output**: `pd.DataFrame`
- **Исключения**: Вызывает `ValueError`, если в столбце 'values' обнаружены значения NaN.

##### 3. `make_dataset(df: pd.DataFrame) -> pd.DataFrame`
- **Описание**: Основная функция для обработки DataFrame. Заполняет пропущенные значения, генерирует признаки и заполняет оставшиеся NaN средним значением столбцов.
- **input**: `pd.DataFrame`
- **output**: `pd.DataFrame`

##### 4. `reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame` (Закомментировано)
- **Описание**: Предназначена для уменьшения использования памяти DataFrame путем приведения числовых типов к более компактным. В настоящее время закомментирована.
- **input**: `pd.DataFrame`
- **output**: `pd.DataFrame`

---

### `src/data/dataset_prep.py` - скрипт для непосредственной подготовки данных.

#### Функции:

##### 1. `load_data() -> Tuple[pd.DataFrame, pd.DataFrame]`
- **Описание**: Загружает обучающий и тестовый наборы данных из файлов формата Parquet. Возвращает кортеж из двух DataFrame.
- **output**: `Tuple[pd.DataFrame, pd.DataFrame]`
- **Исключения**: Возвращает `(None, None)`, если файл не найден.

##### 2. `process_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`
- **Описание**: Обрабатывает обучающий и тестовый наборы данных, применяя функцию `make_dataset` для генерации признаков.
- **input**: `pd.DataFrame`, `pd.DataFrame`
- **output**: `Tuple[pd.DataFrame, pd.DataFrame]`

##### 3. `save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None`
- **Описание**: Сохраняет обработанные обучающий и тестовый наборы данных в файлы формата Parquet после удаления пропущенных значений.
- **input**: `pd.DataFrame`, `pd.DataFrame`
- **output**: `None`

##### 4. `main() -> None`
- **Описание**: Основная функция, которая загружает, обрабатывает и сохраняет наборы данных.
- **output**: `None`

---



Конечно, вот документация для вашего файла `constants.py` в формате, который вы указали:

---

### `src/constants.py` - хранение констант.

#### Константы:

##### 1. `EPOCH_DATE`
- **Описание**: Дата эпохи, используемая в вычислениях.
- **Тип**: `datetime.date`

##### 2. `BENFORD_DIST`
- **Описание**: Распределение Бенфорда, вычисленное для цифр от 1 до 9.
- **Тип**: `np.ndarray`

##### 3. `QUANTILE_25_VALUE` и `QUANTILE_75_VALUE`
- **Описание**: Константы для 25-го и 75-го процентилей.
- **Тип**: `float`

##### 4. `AUTOCORRELATION_LAG`
- **Описание**: Лаг для вычисления автокорреляции.
- **Тип**: `int`

##### 5. `FFT_COEFFICIENT_0` и `FFT_COEFFICIENT_1`
- **Описание**: Коэффициенты для вычисления БПФ (быстрого преобразования Фурье).
- **Тип**: `int`

##### 6. `NUMBER_CROSSING_VALUE`
- **Описание**: Константа для вычисления пересечения чисел.
- **Тип**: `int`

##### 7. `PERMUTATION_ENTROPY_DEFAULT_ORDER` и `PERMUTATION_ENTROPY_DEFAULT_DELAY`
- **Описание**: Константы для вычисления энтропии перестановок.
- **Тип**: `int`

##### 8. `ENERGY_RATIO_DEFAULT_NUM_SEGMENTS` и `ENERGY_RATIO_DEFAULT_SEGMENT_FOCUS`
- **Описание**: Константы для вычисления отношения энергии по сегментам.
- **Тип**: `int`

##### 9. `BEST_PARAMS`
- **Описание**: Гиперпараметры для модели CatBoost.
- **Тип**: `dict`

##### 10. `MODEL_PATH`
- **Описание**: Путь к директории моделей.
- **Тип**: `str`

---

