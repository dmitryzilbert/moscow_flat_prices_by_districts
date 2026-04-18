# Moscow Flat Prices by Districts (Streamlit v2.1)

Streamlit-приложение для интервального анализа цен на квартиры в Москве по районам.

## Что изменилось в v2.1

Добавлен альтернативный режим обработки пропусков по кварталам:

- **Strict exact quarter** — используются только точные значения в выбранных кварталах.
- **Use last available observation up to selected quarter** — для каждого района выбирается последнее доступное наблюдение с `quarter <= selected quarter`.

По умолчанию включен режим carry-forward (`Use last available observation up to selected quarter`).

### Ключевые изменения v2.1

- Добавлены поля selected/effective quarter в snapshot и в таблицы.
- CAGR теперь считается по **фактическому** интервалу `effective_base_quarter -> effective_end_quarter`:
  - `cagr = ((current_price / base_price) ** (4 / effective_delta_quarters) - 1) * 100`.
- `coverage_ratio` и `full_coverage_flag` считаются относительно **effective interval**.
- Tooltip карты показывает selected/effective quarter, delta, цены, сделки, CAGR, pct/abs change, coverage.
- В data quality summary добавлена диагностика:
  - сколько районов имеют точную пару кварталов,
  - сколько районов получили пару только через carry-forward,
  - сколько исключено из-за отсутствия данных до base/end,
  - сколько исключено из-за `effective_end_quarter <= effective_base_quarter`.
- В графике района отмечаются selected и effective кварталы, плюс подпись с ключевыми метриками.

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
- `base_price` = цена в effective базовом квартале
- `current_price` = цена в effective конечном квартале
- `abs_change = current_price - base_price`
- `pct_change = (current_price / base_price - 1) * 100`
- `cagr = ((current_price / base_price) ** (4 / effective_delta_quarters) - 1) * 100`

Где `effective_delta_quarters` — число кварталов между фактически использованными кварталами для конкретного района.

## Прозрачность качества данных

В интерфейсе отображаются:
- число районов в панели,
- число районов с ценой в базе,
- в конце,
- в обоих кварталах,
- прошедших min-deals,
- с полным покрытием effective-интервала,
- реально участвующих в расчете выбранной метрики,
- и отдельные диагностические счетчики strict/carry-forward.

Также считаются:
- `expected_quarters`
- `observed_quarters`
- `coverage_ratio`
- `full_coverage_flag`

## Примечание по preprocessing.py

`preprocessing.py` можно использовать для подготовки `panel.parquet` из CSV.
Приложение v2.1 считает интервальные метрики **на лету** и не зависит от lag-колонок как от основной логики.

## Smoke-test checklist (PowerShell)

1. Перейти в папку проекта: `cd <путь>\moscow_flat_prices_by_districts`.
2. Проверить чтение `panel.parquet`: `python -c "import pandas as pd; df=pd.read_parquet('panel.parquet'); print(df.shape)"`.
3. Проверить чтение `moscow_districts.geojson`: `python -c "import json; print(len(json.load(open('moscow_districts.geojson', encoding='utf-8'))['features']))"`.
4. Запустить приложение: `streamlit run app.py`.
5. Переключать режимы strict/carry-forward и убедиться, что значения пересчитываются.
6. Выбрать метрику `CAGR` и проверить, что карта/таблицы/график обновляются без ошибок.
7. Проверить район без точного `end quarter`: в carry-forward он должен использовать последний доступный квартал.
8. Проверить tooltip карты: должны быть selected/effective quarter и effective delta.
