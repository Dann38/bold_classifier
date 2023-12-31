# Классификатор жирного начертания

Оценка методов в `examples\evaluate.py`

Структура репозитория:
- **types_font** - типы начертания (BOLD, REGULAR)
- **binarizer** - "бинаризатор" - проводит бинаризацию изображения
- **bold_classifier** - "классификатор жирного начертания" (ставит слову в соответствие 
BOLD или REGULAR т.е жирный или не жирный текст)
- **clusterizer** - "кластеризатор по оценки относит к классу"
- **dataset** - пример размеченных данных
- **examples** - примеры работы (evaluate - оценка точности и полноты методов)
- **dataset_reader** - все, что необходимо для чтения изображений и 
получения входных данных (изображение и разметка)

setup загружает 4 пакета: binarizer, bold_classifier, clusterizater, dataset_reader
```python
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

### Классификатор:
**PsBoldClassifier** - отношение периметра к площади (классификация на основе бинаризованного изображения)
*(лучший)*

**MeanBoldClassifier** - среднее значение (классификация на основе бинаризованного изображения)

**MedianBoldClassifier** - медианное значение (классификация на основе бинаризованного изображения)
### Бинариза'тор:
**ValleyEmphasisBinarizer** - бинаризация выделение впадины (адаптивная бинаризация)

**AdapBinarizer** - другой вариант бинаризации (адаптивная бинаризация)


### Кластериза'тор
**Bold2MeanClusterizer** - KMean кластеризация (k=2)

**BoldSpectralClusterizer** - Выделение связных компонент

**BoldFixedThresholdClusterizer** - Пороговая кластеризация (порог 0.4-0.6)