отлично, это уже **готовый проект уровня strong A / research project**.
Ниже — **README**, который ты можешь **прямо целиком скопировать в GitHub**.
Я сделал **двуязычную версию: English + Russian**, кратко, структурировано, без перегруза.

---

# ⭐ MAKHOVA ANASTASIA ⭐

# ML-Enhanced Statistical Arbitrage on Cointegrated Crypto Assets

### (BTC / ETH pairs trading)

---

## 📌 Project Overview

**EN**

This project implements a **market-neutral statistical arbitrage strategy** on a cointegrated crypto pair (Bitcoin and Ethereum), enhanced with **machine learning as a trade filter**.

The strategy combines:

* classical mean-reversion logic,
* econometric formation filtering (half-life),
* data-driven triple-barrier labeling,
* interpretable machine learning (logistic regression),
* dynamic position sizing and risk control.

The goal is to achieve **stable risk-adjusted returns** while maintaining low correlation with the underlying market.

---

**RU**

В проекте реализована **рыночно-нейтральная стратегия статистического арбитража** для пары криптоактивов BTC–ETH с использованием **машинного обучения как фильтра сделок**.

Стратегия объединяет:

* классический mean-reversion,
* формационный фильтр на основе half-life,
* трипл-барьерную разметку,
* интерпретируемую ML-модель (логистическая регрессия),
* динамический сайзинг и риск-контроль.

Цель — получить **устойчивую доходность с учетом риска** при минимальной зависимости от рынка.

---

## 📊 Data

**EN**

* Assets: **BTC-USD, ETH-USD**
* Frequency: **hourly**
* Period: **~2 years**
* Source: Yahoo Finance
* Prices: adjusted close, synchronized timestamps

Hourly crypto data is well-suited due to 24/7 trading and absence of overnight gaps.

**RU**

* Активы: **BTC и ETH**
* Частота: **1 час**
* Период: **около 2 лет**
* Источник: Yahoo Finance
* Используются синхронизированные скорректированные цены

---

## 📐 Spread Construction

**EN**

The strategy operates on the log-price spread:

<math display="block">
s_t = \log(P^{BTC}_t) - \log(P^{ETH}_t)
</math>

The spread is standardized into a rolling **z-score**, which serves as the core trading signal.

**RU**

Стратегия работает со спредом лог-цен:

<math display="block">
s_t = \log(P^{BTC}_t) - \log(P^{ETH}_t)
</math>

Далее он нормализуется в **z-score**, используемый для входов и выходов.

---

## ⏱ Mean Reversion & Formation Filter

**EN**

Mean reversion strength is measured via **half-life**, estimated from an AR(1) model:

<math display="block">
\Delta s_t = \lambda s_{t-1} + \varepsilon_t, \quad
HL = -\frac{\ln 2}{\lambda}
</math>

Trading is allowed only when half-life is below a predefined threshold, ensuring stable mean-reverting regimes.

**RU**

Скорость возврата к среднему измеряется через **half-life**, оцененный из AR(1):

<math display="block">
HL = -\frac{\ln 2}{\lambda}
</math>

Торговля разрешена только в режимах с быстрым mean-reversion.

---

## 🧠 Machine Learning Layer

**EN**

Machine learning is used **only as a trade filter**, not as a signal generator.

* Model: **Logistic Regression**
* Features:

  * |z|-score
  * Δz
  * local spread volatility
  * half-life
* Labels: **triple-barrier method**
* Applied strictly **out-of-sample**

ML improves robustness by filtering low-quality mean-reversion signals.

**RU**

Машинное обучение используется **только как фильтр сделок**.

* Модель: логистическая регрессия
* Признаки:

  * величина отклонения (|z|)
  * динамика (Δz)
  * локальная волатильность
  * half-life
* Разметка: трипл-барьер
* Применение: только на test-периоде

---

## 🚧 Triple-Barrier Labeling

**EN**

Each trade candidate is labeled based on:

* profit-taking barrier,
* stop-loss barrier,
* time-based exit.

Barriers are **data-driven**, calibrated via normalized:

* Maximum Favorable Excursion (MFE),
* Maximum Adverse Excursion (MAE).

This ensures realistic, path-dependent supervision.

**RU**

Каждая потенциальная сделка размечается по:

* take-profit,
* stop-loss,
* временной границе.

Границы калибруются **эмпирически** через MFE и MAE, нормализованные на волатильность.

---

## 📈 Trading Logic

**EN**

* Enter when |z| exceeds entry threshold
* Exit on:

  * mean reversion,
  * stop-loss,
  * max holding time,
  * market shock
* Position size scales with |z|
* Cooldown after exits
* Shock filter blocks extreme market events

**RU**

* Вход при сильном отклонении z-score
* Выход по:

  * возврату к среднему,
  * стопу,
  * лимиту времени,
  * рыночным шокам
* Размер позиции зависит от силы сигнала

---

## 📊 Performance Metrics

**EN**

* Total PnL
* Total Return (%)
* Sharpe Ratio (annualized)
* Turnover
* Fraction Tradable
* Maximum Drawdown

Benchmarks:

* BTC Buy & Hold
* ETH Buy & Hold
* BTC–ETH Spread Buy & Hold

**RU**

* Совокупный PnL
* Доходность
* Sharpe ratio
* Оборот
* Доля торгуемых периодов
* Максимальная просадка

Сравнение с buy-and-hold бенчмарками.

---

## ✅ Key Findings

**EN**

* Strategy achieves strong risk-adjusted returns
* Low correlation with BTC market
* Significantly smaller drawdowns than buy-and-hold
* ML improves robustness depending on formation update frequency
* Results stable across train/test split

**RU**

* Высокая доходность с учетом риска
* Рыночно-нейтральное поведение
* Существенно меньшие просадки
* ML повышает устойчивость стратегии
* Сопоставимые результаты на train и test

---

## 📚 References

* Avellaneda & Lee (2010) — *Statistical Arbitrage in the US Equities Market*
* Gatev, Goetzmann, Rouwenhorst (2006) — *Pairs Trading*
* López de Prado (2018) — *Advances in Financial Machine Learning*
* Ernest P. Chan — *Algorithmic Trading*


