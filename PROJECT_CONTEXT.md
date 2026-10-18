# PROJECT CONTEXT :: OPEN-QUANT-SPRINT (SYNDICATE)

> **Назначение документа**: Единый источник правды (Single Source of Truth) по архитектуре, методологии, кодовой базе, социальной стратегии и бизнес-модели проекта `open-quant-sprint`. Документ спроектирован так, чтобы любая языковая модель (LLM) в новой сессии могла мгновенно восстановить 100% контекста проекта и продолжить работу без галлюцинаций.

---

## 1. Executive Summary & Философия

### 1.1. Проблема на рынке
Индустрия студенческого и ритейл-квант-трейдинга переполнена двумя вещами:
1. **Платные фейк-стажировки ($50–$500)**: студентам продают тривиальные Jupyter-ноутбуки, шаблонные задачи без ревью и бесполезные PDF-сертификаты.
2. **Сломанные бэктесты в соцсетях**: 90%+ графиков доходности в LinkedIn/X содержат грубые ошибки:
   - **Look-Ahead Bias**: нормализация цен по всей выборке (`df.mean()`, `df.std()`), заглядывание вперед (`shift(-1)`), утечка данных в сглаживателях (двусторонние фильтры Савицкого-Голея, центрированные скользящие средние).
   - **Нулевые издержки**: отсутствие комиссий (taker fee 3–5 bps), проскальзывания и фандинга.
   - **Нереалистичный оборот (Turnover)**: стратегия ребалансирует 100% портфеля каждый час и мгновенно сгорает в реальном стакане.
   - **Overfitting**: выбор параметров под один исторический кусок без Out-of-Sample и Purged Cross-Validation.

### 1.2. Решение: Open Quant Sprint
Двухнедельный открытый спринт для разработчиков и студентов-квантов вокруг публичного репозитория:
- **Никакой инфоцыганщины**: бесплатно, открытый код, отбор по реальным навыкам написания Python/NumPy/Pandas кода (anti-vibecoding).
- **Готовая инфраструктура**: реальный датасет Binance Futures (20 бессрочных контрактов, 1h бары, фандинг, объемы тейкеров).
- **Автоматический аудит-харнесс (`src/audit.py`)**: жесткая проверка на утечки данных префиксным усечением, учет задержки исполнения в 1 бар, учет фандинга и кривая деградации от комиссий (0–10 bps).
- **Public & Private Holdout**: публичный `train_1h.parquet` для исследований и закрытый `holdout_1h.parquet` (хранится локально у организатора), открываемый только в финале спринта для независимой верификации.
- **Результат**: каждый участник получает рабочий публичный PR в репозитории с честным аудиторским отчетом, на который можно ссылаться на собеседованиях в проп-фирмы и хедж-фонды.

---

## 2. Репозиторий и Структура Кодовой Базы

- **GitHub Remote**: `https://github.com/valiyevoktay-cmd/open-quant-sprint.git`
- **Локальный путь**: `d:\syndicate`
- **Стек**: Python 3.10+, NumPy, Pandas 2.0+, SciPy, PyArrow, Pytest. Без тяжелых фреймворков.

```
d:\syndicate\
├── .gitignore               # Игнорирует кэши и data/holdout_*.parquet
├── pyproject.toml           # Метаданные пакета open-quant-sprint и настройки pytest
├── README.md                # Предельно сухой технический ридми (спецификация, запуск, аудит)
├── PROJECT_CONTEXT.md       # Этот документ (полный мастер-контекст)
├── data/
│   ├── build_dataset.py     # Парсер REST API Binance Futures + калиброванный генератор
│   ├── train_1h.parquet     # Публичный тренировочный датасет (20 пар, 69 120 строк)
│   └── holdout_1h.parquet   # ПРИВАТНЫЙ сейф-датасет (gitignored, 17 280 строк)
├── src/
│   ├── __init__.py
│   ├── schema.py            # Каноническая схема 11 колонок, парсинг свечей, Dollar Bars
│   ├── quality.py           # Аудит качества (DQS 0-100), автоочистка и восстановление OHLC
│   ├── features.py          # Taker Flow Imbalance, Parkinson Vol, FFD, Cross-Sectional Z, Deadband
│   ├── stats.py             # ADF-тест, Deflated Sharpe Ratio (DSR), векторизованный Rank IC
│   ├── backtest.py          # Векторизованный бэктестер с задержкой в 1 бар, фандингом и комиссиями
│   └── audit.py             # Харнесс: проверка каузальности, Purged IS/OOS, Holdout, CLI-отчет
├── strategies/
│   ├── template.py          # Базовый шаблон для участников (24h reversal)
│   ├── baseline_flow.py     # Рабочий бейзлайн (Taker Flow + Breakout + Funding + Deadband)
│   └── leaky_example.py     # Намеренно сломанная стратегия для проверки поимки ликов
└── tests/
    └── test_suite.py        # 9 групп интеграционных, стресс и пермутационных тестов (Pytest)
```

---

## 3. Спецификация Данных и Схемы

### 3.1. Канонические колонки (`src.schema.REQUIRED_COLS`)
Каждый датасет гарантированно содержит 11 типизированных колонок:
1. `timestamp` (`datetime64[ns, UTC]`): метка открытия часового бара в UTC.
2. `symbol` (`str`): тикер бессрочного фьючерса (например, `BTCUSDT`, `ETHUSDT`).
3. `open` (`float64`): цена открытия.
4. `high` (`float64`): максимальная цена.
5. `low` (`float64`): минимальная цена.
6. `close` (`float64`): цена закрытия.
7. `volume` (`float64`): базовый объем в монетах.
8. `quote_volume` (`float64`): объем в USDT.
9. `taker_buy_quote_volume` (`float64`): агрессивный объем покупок по рынку (в USDT).
10. `trades` (`int64`): количество сделок за час.
11. `funding_rate` (`float64`): ставка финансирования бессрочного контракта (начисляется раз в 8 часов, в остальные часы `0.0`).

### 3.2. Набор активов (20 liquid perpetuals)
`DEFAULT_SYMBOLS`: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `DOTUSDT`, `NEARUSDT`, `LTCUSDT`, `UNIUSDT`, `ATOMUSDT`, `ARBUSDT`, `OPUSDT`, `APTUSDT`, `SUIUSDT`, `INJUSDT`, `AAVEUSDT`.

### 3.3. Разделение Train / Holdout
- **`data/train_1h.parquet`**: первые 80% временной шкалы (3 456 часов $\times$ 20 символов = 69 120 строк). Открыт для всех участников. Оценка качества данных: **DQS = 99.93 / 100 (READY)**.
- **`data/holdout_1h.parquet`**: последние 20% временной шкалы (864 часа $\times$ 20 символов = 17 280 строк). Строго скрыт в `.gitignore`, используется организатором для финальной проверки на переподгонку.

---

## 4. Математика, Метрики и Архитектура Харнесса

```
                         [ Участник: compute_weights(df) ]
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       check_causality_truncation (src/audit.py)      │
             │   - Изоляция: df.copy() (защита от мутаций)         │
             │   - Детерминизм: 2 прогона дают w1 == w2            │
             │   - Срезы 40%, 65%, 85%: max|w_trunc - w_full| < 1e-9│
             │   - Отсев нулевых/дегенеративных стратегий          │
             └──────────────────────────┬──────────────────────────┘
                                        │ PASS
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │            Vectorized Backtest Engine               │
             │   - Нормализация: sum(|w_t|) <= 1.0                 │
             │   - 1-Bar Lag: позиция w_{t-1} зарабатывает ret_t    │
             │   - Фандинг: - (w_{t-1} * funding_rate_t)           │
             │   - Комиссии: turnover_t * fee_rate (0, 3, 5, 10bps)│
             └──────────────────────────┬──────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  ┌───────────────────────────────┐             ┌───────────────────────────────┐
  │   Purged In/Out of Sample     │             │    Statistical Validation     │
  │   - 70% In-Sample             │             │   - Vectorized Rank IC (1h)   │
  │   - 24h Embargo Buffer        │             │   - Deflated Sharpe Ratio     │
  │   - 30% Out-of-Sample @ 5 bps │             │   - ADF Stationary / Parkinson│
  └───────────────────────────────┘             └───────────────────────────────┘
```

### 4.1. Проверка на утечки будущего (Prefix-Cut Causality)
Функция `check_causality_truncation` в [`src/audit.py`](file:///d:/syndicate/src/audit.py):
1. Прогоняет `compute_weights` на полном датасете и получает $w_{full}$.
2. Делает префиксные срезы временной шкалы на чекпоинтах 40%, 65% и 85% ($T_k$).
3. Прогоняет `compute_weights` на изолированном срезе $[0 \dots T_k]$.
4. Сравнивает полученные веса $w_{trunc}$ с соответствующими значениями $w_{full}[0 \dots T_k]$.
5. Если $\max |w_{trunc} - w_{full}| > 10^{-9}$, фиксируется утечка данных (Look-ahead bias), и аудит мгновенно прерывается с ошибкой `[FAIL] Look-ahead leak detected`.

### 4.2. Исполнение и учет издержек (`src/backtest.py`)
- **Задержка исполнения в 1 бар**: сигнал формируется на закрытии бара $t-1$. В сделку входим на открытии бара $t$. Доходность позиции равна $w_{t-1} \cdot \left(\frac{\text{close}_t}{\text{close}_{t-1}} - 1\right)$.
- **Фандинг**: длинная позиция платит положительный фандинг, короткая — получает: $-\, w_{t-1} \cdot \text{funding\_rate}_t$.
- **Оборот и комиссии**:
  $$\text{turnover}_t = \sum_{i} |w_{i, t} - w_{i, t-1}|, \quad \text{fee\_cost}_t = \text{turnover}_t \cdot \text{fee\_rate}$$
  Аудит тестирует сетку комиссий: `0.0 bps` (теория), `3.0 bps` (VIP/мейкер), `5.0 bps` (стандартный тейкер Binance Futures), `10.0 bps` (стресс/проскальзывание).

### 4.3. Очищенное OOS-разбиение (Purged IS/OOS Split)
Чтобы исключить перенос автокорреляции между выборками, между In-Sample (первые 70%) и Out-of-Sample (последние 30%) вставляется буфер эмбарго в 24 часа (`embargo_bars = 24`). Оценка OOS производится строго при комиссии `5.0 bps`.

### 4.4. Статистическая валидация
- **Deflated Sharpe Ratio (DSR)** по Бейли и Лопесу де Прадо (2014): корректирует коэффициент Шарпа на асимметрию (skewness), эксцесс (kurtosis) и множественное тестирование ($N = 10$ попыток). Показывает вероятность того, что найденный Шарп не является ложноположительным.
- **Векторизованный Rank IC**: кросс-секционная корреляция Спирмена между весами $w_t$ и форвардной доходностью $r_{t+1}$. Рассчитывается через `groupby("timestamp").rank(method="average")` за ~15 мс на всем датасете.
- **Волатильность Паркинсона**:
  $$\sigma_P = \sqrt{\frac{1}{4 \ln(2) \cdot N} \sum_{t=1}^N \left(\ln\frac{H_t}{L_t}\right)^2}$$
  Каузальный роллинг-эстиматор диапазонной волатильности, более эффективный, чем оценка по ценам закрытия.
- **Фильтр мертвой зоны (`rebalance_deadband`)**:
  Эвристика подавления шумного оборота. Обновляет позицию по монете $i$ только тогда, когда целевой вес изменился сильнее, чем `buffer` ($|w_{target} - w_{current}| > 0.012$). Снижает часовой оборот в 5–6 раз без потери альфы.

### 4.5. Стресс-тест чувствительности к задержке (Lag Sensitivity Test)
Проверяет поведение стратегии при искусственном сдвиге исполнения еще на 1 бар позже (`lag_bars = 2` vs `lag_bars = 1`):
- Настоящий фундаментальный/микроструктурный сигнал деградирует плавно (допустимо удержание $\ge 0\%$ Шарпа).
- Синтетический лик (Look-ahead leak) полностью схлопывается (обвал Шарпа на 95–100%).

### 4.6. Тест нулевой гипотезы (Shuffled Returns Permutation)
Случайно перемешивает форвардные доходности активов по всей панели при сохранении структуры весов портфеля:
- Исключает ложноположительную альфу, создаваемую артефактами обработки данных, дрейфом цен или ошибками расчета PnL.
- Порог прохождения: абсолютный Шарп на перемешанных данных $|SR_{null}| < 0.75$.

### 4.7. Аудит регулярности фандинга (Point-in-Time Funding Audit)
- Бессрочные фьючерсы Binance рассчитывают ставку финансирования строго каждые 8 часов (00:00, 08:00, 16:00 UTC).
- Проверка `funding_irregularities` гарантирует, что ненулевой фандинг не начисляется в промежуточные часы (защита от сдвига таймстемпов при объединении таблиц).

### 4.8. Декомпозиция издержек (Friction Drag Breakdown)
В отчет бэктеста и CLI-аудита выводится явная декомпозиция кумулятивной доходности:
$$\text{Net PnL} = \text{Gross PnL} - \text{Fee Cost} - \text{Funding Cost}$$
Позволяет сразу видеть, убивают ли стратегию комиссии тейкера или отрицательный перенос ставок финансирования.

---

## 5. Эталонные Стратегии в Репозитории

| Стратегия | Логика | Causality Check | Rank IC (1h) | Turnover | Sharpe @ 0bps | Sharpe @ 5bps | Holdout Sharpe | Вердикт |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`leaky_example.py`](file:///d:/syndicate/strategies/leaky_example.py) | Глобальный `mean()` + `shift(-1)` | **FAIL** (diff > 0.61) | -0.0255 | 1.3918 | -2.11 | -13.66 | - | **[REJECT]** |
| [`template.py`](file:///d:/syndicate/strategies/template.py) | Наивный 24h реверсал без сглаживания | **PASS** (diff = 0) | +0.0124 ($p=0.01$) | 0.2925 | -3.99 | -8.28 | - | **[REJECT]** |
| [`baseline_flow.py`](file:///d:/syndicate/strategies/baseline_flow.py) | Taker Flow + Breakout + Funding + Deadband | **PASS** (diff = 0) | +0.0003 | 0.0491 | **+3.37** | **+1.94** | **+2.00** | **[PASS]** |

### Чему учит тройка примеров:
1. `leaky_example.py` показывает, как харнесс отлавливает тонкие ошибки заглядывания в будущее, которые студенты часто не замечают.
2. `template.py` показывает классическую ловушку: сигнал имеет статистическую значимость ($t = +2.52$), но из-за гигантского оборота (29% в час) и волатильных пробоев моментально сливает депозит на комиссиях.
3. `baseline_flow.py` демонстрирует профессиональный подход: нормирование на волатильность Паркинсона, штраф за скученность через ставку финансирования, каузальное сглаживание и мертвую зону ребалансировки.

---

## 6. Социальная Стратегия и Воронка Спринта

```
[ Пост на Reddit (r/algotrading, r/quant) ]
                      │
                      ▼
[ Комментарии в треде (алгоритмический буст) ]
                      │ Личное сообщение (DM)
                      ▼
[ Краткая Google/Tally Форма (2 мин, фильтр на Python/Math) ]
                      │
                      ▼
[ Допуск в Discord/Telegram спринта + ссылка на GitHub ]
                      │ 14 дней спринта
                      ▼
[ PR с кодом стратегии + Автоматический аудит CI ]
                      │ Финал: разблокировка Holdout Vault
                      ▼
[ Топ-участники -> Приглашение в закрытый Синдикат ]
```

### 6.1. Reddit Пост (Финальный согласованный текст)
- **Целевые сабреддиты**: `r/algotrading`, `r/quant`, `r/CryptoCurrency`.
- **Заголовок**:
  `Anyone down for a 2-week quant sprint on real crypto data? (Stress-testing signals, leak audits, fees)`
- **Тело поста**:
  ```text
  Hey, Reddit

  We are a small team of students from Sweden and the UK. Spent the last year mostly writing backtests, playing with market data, and trying to implement papers.

  Seems like half the student quant stuff out there is either people paying $50 for fake 'research internships', or posting backtests on LinkedIn that only look good.

  Just looking for people to collaborate with for a couple of weeks instead of running backtests in isolation.

  The idea is just to build a few basic signals, review each other's code for data leaks, and stress-test them with actual slippage and fees. Whatever works or breaks goes straight into a shared public repo so everyone has something real to point to in interviews.

  We already sorted out the plumbing: cleaned 1h Binance futures data for 20 pairs (funding rates and taker flow included) and a test script that catches look-ahead leaks and runs your strategy with 1-bar delay and taker fees.

  Looking for people who actually know Python and basic stats (not vibecode pls). 

  The sprint itself is 2 weeks, but if we click and work well together, we'd definitely be down to keep collaborating on bigger projects afterwards.

  Drop a comment below if you’re interested, and I’ll DM you the repo and a quick 2-minute form so we can get everyone organized.
  ```

### 6.2. LinkedIn Пост (Конструктивная инженерная подача)
Публиковать через 24–48 часов после старта Reddit-ветки.
- **Тон**: Инженерная рефлексия, открытая инициатива, привлечение внимания рекрутеров.
- **Текст**:
  ```text
  Why do 90% of quantitative backtests fail when deployed in production?

  Over the past year, while implementing academic papers and studying market microstructure across perpetual futures, we noticed two consistent issues in student/retail quant research:
  1. Subtle look-ahead bias: in-sample standardization, lookahead rolling windows, and data leakage across time boundaries.
  2. The cost-decay cliff: strategies that look impressive on zero fees completely collapse once subjected to a realistic 1-bar execution delay, 8-hour funding rate debits, and 5 bps taker fees.

  To solve this for ourselves and help other students build verifiable work, our team (Sweden & UK) put together an open-source research sprint harness:
  - Standardized 1h panel across 20 liquid Binance perpetual futures pairs (including taker flow imbalance and funding rates).
  - An automated causal validation engine that proves temporal invariance via prefix-cut truncation.
  - Purged In-Sample / Out-of-Sample evaluation with turnover control.

  We are running a focused 2-week collaborative sprint where participants implement alpha signals, peer-review PRs for data leaks, and stress-test models under institutional friction. All verified work goes into an open public repository.

  If you have a strong background in Python/statistics and want to collaborate on real data, feel free to drop a comment or reach out via DM.

  #QuantFinance #AlgorithmicTrading #MachineLearning #Python #OpenSource
  ```

### 6.3. Квалификационная форма участника (Google Forms / Tally, 2 минуты)
Вопросы для отсева нерелевантной аудитории:
1. GitHub профиль (обязательно).
2. Опыт в Python / Pandas (шкала 1–5 + ссылка на репозиторий или проект).
3. Что такое Look-Ahead Bias и как его избежать в скользящих средних? (1–2 предложения, проверка на понимание каузальности).
4. Сколько часов в неделю готов выделить на двухнедельный спринт? (Норма: 5–10 часов).
5. Никнейм в Discord / Telegram для добавления в рабочую группу.

---

## 7. Регламент и Ритм 2-Недельного Спринта

```
Дни 1-3: Онбординг & Сетап
├── Клонирование open-quant-sprint
├── Прогон pytest и src.audit на baseline_flow.py
└── Выбор альфа-гипотезы (моментум, реверсал, микроструктура, фандинг)

Дни 4-8: Разработка & Кросс-Ревью
├── Написание strategies/<name>.py
├── Локальный прогон python -m src.audit strategies/<name>.py
└── Оформление Pull Request и взаимное ревью кода на утечки данных

Дни 9-12: Оптимизация Издержек & OOS Валидация
├── Внедрение rebalance_deadband для подавления оборота
├── Тестирование на сетке комиссий (3, 5, 10 bps)
└── Прохождение 70/30 Purged Split (OOS Sharpe > 0 @ 5 bps)

Дни 13-14: Финал & Holdout Vault
├── Организатор прогоняет PR через скрытый data/holdout_1h.parquet
├── Публикация итоговой лидерборд-таблицы
└── Мёрдж выживших стратегий в main ветку публичного репо
```

---

## 8. Долгосрочная Модель Монетизации Синдиката

Спринт — это верх воронки по отбору лучших талантов. Дальнейшее развитие синдиката строится вокруг пула проверенных некастодиальных стратегий.

### 8.1. Форматы Монетизации
1. **Проп-трейдинговые челленджи и аллокации (Prop Firm Evaluation)**:
   - Объединение выживших стратегий в диверсифицированный ансамбль с ограничением максимальной просадки (Max Daily DD < 4%, Total DD < 8%).
   - Прохождение челленджей на аллокации капитала ($100k–$400k) с разделением прибыли 80/20 в пользу синдиката.
2. **Лицензирование сигналов и API для внешних фондов**:
   - Предоставление доступа к API потоку весов портфеля институциональным клиентам за фиксированную абонентскую плату ($1,500–$5,000/мес) + % от alpha-profit.
3. **PMA (Professional Managed Accounts) / Субаккаунты**:
   - Торговля на биржевых субаккаунтах инвесторов (Binance Corporate, Bybit Institutional, OKX) через API-ключи без права вывода средств.
   - Механика: High-Water Mark, ежемесячный Performance Fee (20–25%).
4. **Венчурный синдикат / SPV (Special Purpose Vehicle)**:
   - Юрисдикция: Delaware LLC (США) или BVI / Кайманы для крипто-структур.
   - Участники синдиката получают доли в пуле в зависимости от вклада их альфа-моделей в общий PnL (Shapley Value Attribution).

### 8.2. Защита Интеллектуальной Собственности (IP)
- Публичный репозиторий содержит **бейзлайны, инфраструктуру и архитектуру аудита**. Это витрина инженерной компетенции.
- Продакшн-ансамбли и веса сложных моделей авторов оборачиваются в приватные модули, защищенные договором участника синдиката (IP Contributor Agreement с фиксацией авторского роялти).

---

## 9. Руководство для AI-Ассистентов (Правила работы с кодом)

Если ты — AI-ассистент, работающий с этим проектом, ты **ОБЯЗАН** соблюдать следующие правила:

1. **Никакого Look-Ahead Bias**:
   - Никогда не вызывай глобальные функции `.mean()`, `.std()`, `min()`, `max()` по временной оси без `.rolling()` или каузального `.ewm()`.
   - Никогда не используй отрицательные сдвиги `.shift(-k)` при расчете сигналов или весов.
   - Любая кросс-секционная нормализация (`cross_sectional_zscore`) должна выполняться строго внутри конкретного `timestamp`.
2. **Всегда помни про 1-Bar Lag и издержки**:
   - Любая новая стратегия обязана выдерживать комиссию минимум `5.0 bps` при исполнении со сдвигом на 1 бар.
   - Обязательно контролируй оборот (Turnover). Используй `rebalance_deadband(df, weights, buffer=...)` для подавления лишних сделок.
3. **Не трогай приватный Holdout**:
   - Файл `data/holdout_1h.parquet` должен оставаться в `.gitignore`. Никогда не пытайся коммитить его или обучать параметры стратегий на его данных.
4. **Стиль кода — Не ИИ-шный (No AI Slop)**:
   - Пиши идиоматичный, лаконичный Python/Pandas/NumPy код, как профессиональный quant developer.
   - Никаких декоративных эмодзи в логах, CLI и комментариях кода.
   - Никаких пустых шаблонных классов-оберток ради видимости архитектуры. Только чистые, быстрые функции и dataclass'ы.
5. **Валидация любых изменений**:
   - После любого изменения в `src/` или `strategies/` обязательно запускай тесты:
     ```bash
     python -m pytest -v
     ```
   - Запускай аудит бейзлайна:
     ```bash
     python -m src.audit strategies/baseline_flow.py --holdout data/holdout_1h.parquet
     ```
   - Убедись, что аудит выводит `[PASS] SURVIVED AUDIT`.
