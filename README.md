Machine Learning project | TimeSeries binary classification.
=============================================================

Описание проекта.

---
Организация проекта
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
    │   └──models         <- Скрипты для обучения моделей и использования обученных моделей
    │       │                 для предсказаний
    │       ├── predict_model.py
    │       └── train_model.py
    └── tox.ini            <- tox файл с настройками для запуска tox; см. tox.readthedocs.io
```

--------
