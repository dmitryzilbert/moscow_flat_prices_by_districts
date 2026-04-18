# Moscow Flat Prices by Districts (Streamlit v2)

Streamlit-приложение для интервального анализа цен на квартиры в Москве по районам.

## Что изменилось в v2

Новая логика: **базовый квартал → конечный квартал** (вместо «конечный квартал + окно назад»).

Пользователь выбирает:
- базовый квартал,
- конечный квартал,
- метрику (CAGR / рост % / абсолютное изменение / цена в базе / цена в конце),
- фильтры качества данных (min deals, покрытие интервала).

## Входные файлы

Приложение запускается локально из двух файлов в корне проекта:
- `panel.parquet`
- `moscow_districts.geojson`

Минимально ожидаемые колонки в `panel.parquet`:
- `district_name`
- `quarter` **или** (`year`, `q_num`)
- `price_per_m2`
- `n_deals`

## Запуск

```bash
streamlit run app.py
```

## Метрики

Для выбранного интервала (base → end):
- `base_price` = цена в базовом квартале
- `current_price` = цена в конечном квартале
- `abs_change = current_price - base_price`
- `pct_change = (current_price / base_price - 1) * 100`
- `cagr = ((current_price / base_price) ** (4 / delta_quarters) - 1) * 100`

Где `delta_quarters` — число кварталов между базовым и конечным.

## Прозрачность качества данных

В интерфейсе отображаются:
- число районов в панели,
- число районов с ценой в базе,
- в конце,
- в обоих кварталах,
- прошедших min-deals,
- с полным покрытием,
- реально участвующих в расчете выбранной метрики.

Также считаются:
- `expected_quarters`
- `observed_quarters`
- `coverage_ratio`
- `full_coverage_flag`

## Примечание по preprocessing.py

`preprocessing.py` можно использовать для подготовки `panel.parquet` из CSV.
Приложение v2 считает интервальные метрики **на лету** и не зависит от lag-колонок как от основной логики.


## Smoke-test checklist (PowerShell)

1. Перейти в папку проекта: `cd <путь>\moscow_flat_prices_by_districts`.
2. Проверить чтение `panel.parquet`: `python -c "import pandas as pd; df=pd.read_parquet('panel.parquet'); print(df.shape)"`.
3. Проверить чтение `moscow_districts.geojson`: `python -c "import json; print(len(json.load(open('moscow_districts.geojson', encoding='utf-8'))['features']))"`.
4. Запустить приложение: `streamlit run app.py`.
5. В UI выбрать разные пары `base quarter` / `end quarter` и убедиться, что значения пересчитываются.
6. Выбрать метрику `CAGR` и проверить, что карта/таблицы обновляются без ошибок.
7. Проверить, что таблицы `top` и `bottom` строятся для выбранного интервала.
8. Выбрать конкретный район и убедиться, что график района строится.
9. Проверить кейс отсутствия данных по району в интервале: район серый на карте, приложение не падает.
