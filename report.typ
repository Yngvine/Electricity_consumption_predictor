
#align(center)[
  #v(2cm)
  #text(size: 24pt, weight: 700)[Final Assignment Report]
  #v(6pt)
  #text(size: 14pt)[Master's in Machine Learning 2025]
  #v(1.2cm)
  #text(size: 12pt)[
    Author: Igor Vons, Endika Aguirre and Maria Ines Haddad

    Date: April 12, 2026
  ]

]

#pagebreak()

= Problem Statement
The objective of this project is to forecast the hourly electricity demand (`total load actual` in MW) for the Spanish national grid. Accurate load forecasting is critical for grid stability, energy trading, and efficient resource allocation.

= Dataset
The analysis is based on a four-year hourly dataset spanning from January 1, 2015, to December 31, 2018 (approximately 35,000 observations).

== Source
The data integrates two main sources:
- *Energy Data*: Sourced from the ENTSO-E Transparency Platform, detailing total load, load forecasts, and various electricity generation capacities.
- *Weather Data*: Sourced from the OpenWeatherMap API, containing historical weather features averaged across five major Spanish cities (Barcelona, Bilbao, Madrid, Seville, and Valencia).

== Quality
Overall data quality is high, with some structural missingness in specific generation columns tracking sources that are either absent or unrepresented in the Spanish grid (fully-empty columns were dropped). 

For the target variable (`total load actual`), only 36 hourly records (about 0.1%) were missing. An imputation experiment compared multiple candidate fill strategies. The official day-ahead `total load forecast` was selected as the optimal proxy ($R^2 = 0.990$, MAE $approx$ 316 MW) over linear time-interpolation or the sum of individual generation sources (which systematically under-counted demand). Although this imputation method introduces a small amount of bias from the forecast, the small proportion of missing data and the high quiality of the day-ahead forecast justify this choice.

#figure(
  image("resources/imputation_comparison.png"),
  caption: [Comparison of target imputation strategies. Left: Scatter plot of proxy estimates vs. actual load, showing the day-ahead forecast tightly tracking the perfect-fit diagonal. Right: Residual density distributions, highlighting the minimal error (MAE $approx$ 316 MW) of the forecast approach.]
)

= Exploratory Data Analysis

== Seasonal Patterns
Electricity demand exhibits strong, multi-scale seasonal cycles:
- *Intra-day*: A clear double-peak structure with daily maximums occurring around 09:00 and 20:00, and a significant trough during the night.
- *Intra-week*: Working days (Monday to Friday) carry a 15–20% higher base load compared to weekends.
- *Annual*: Demand peaks during winter and autumn periods.

#figure(
  image("resources/mean_values.png"),
  caption: [Box plots illustrating the distribution of electrical load (MW). Left: Hourly distribution. Center: Daily distribution. Right: Monthly distribution, highlighting daily demand cycles and seasonal variations.]
)

#figure(
  image("resources/grid_laod_comparison.png", width: 80%),
  caption: [Heatmap of median electrical load (MW) by hour and day of the week, illustrating the sustained peak demand during weekday working hours compared to overall lower weekend usage.]
)

== External Factors and Correlation
Weather heavily influences the grid load, primarily through a non-linear relationship with temperature. Exploratory plots reveal a distinct U-shape curve, indicating that extreme temperatures (both cold spells for heating and heatwaves for cooling) drive higher electricity demand.

#figure(
  image("resources/feature_bicorelation.png"),
  caption: [Scatter plots analyzing the impact of weather variables on grid operations. The panels illustrate the U-shaped, non-linear relationship between temperature and electrical load (left), the weak correlation between humidity and load (center), and the highly variable output of onshore wind generation relative to average city wind speeds (right).]
)

== Autocorrelation and Stationarity
Decomposition and autocorrelation analysis (ACF/PACF) on the primary hourly resolution confirm the presence of dominant seasonal periods at lags 24 (daily cycle) and 168 (weekly cycle). Short-term persistence is also extremely strong.

#figure(
  image("resources/trend_decomposition.png"),
  caption: [Decomposition of the hourly electrical load series into trend, seasonal, and residual components. The top panel shows the original series with a clear upward trend and strong seasonal cycles. The middle panel isolates the seasonal component, highlighting the consistent daily and weekly patterns. The bottom panel displays the residuals, which appear stationary and suitable for further modeling.]
)

Confirmatory checks using both ADF (Augmented Dickey-Fuller) and KPSS tests on the primary hourly series indicate statistical stationarity overall. However, applying regular differencing ($d=1$) and seasonal differencing ($D=1, s=24$) heavily cleans the short-memory structure, making the series highly suitable for classical lag-based modelling techniques.
#figure(
  image("resources/hour_lags.png", width: 100%),
  caption: [ACF and PACF plots displaying a clear 24h seasonal pattern with significant spikes.]
)
#figure(
  image("resources/week_lags.png", width: 100%),
  caption: [ACF and PACF plots displaying a clear 168h (weekly) seasonal pattern with significant spikes.]
)
#figure(
  image("resources/year_lags.png", width: 100%),
  caption: [ACF and PACF plots displaying a clear annual (52-week) seasonal pattern with significant spikes.]
)

== Seasonal Cleaning Check on the Transformed Series

To verify that the main daily seasonal structure was effectively extracted, we inspected ACF/PACF after applying joint differencing: regular differencing ($d=1$) and seasonal differencing ($D=1, s=24$). The transformed diagnostics show a much flatter dependence profile, with most correlations oscillating close to zero outside the first short lags.

#figure(
	image("resources/clean_hour_lags.png", width: 100%),
	caption: [ACF and PACF of the transformed hourly load after applying $d=1$ and $D=1$ with $s=24$. The dominant daily seasonal signature is strongly attenuated, confirming effective seasonal extraction.]
)

The key interpretation is that the strong seasonal peaks seen in the raw series are no longer dominant in the transformed space, which is consistent with correct seasonality cleaning. There are still mild structured spikes around multiples of 24 (especially in PACF), but they are substantially smaller.

This reading also supports the lag design used in the feature-based models. Lag 1 captures immediate inertia, while lags 24 and 48 retain informative intra-day recurrence that may remain after transformation due to load-shape asymmetries and calendar effects. Lag 168 remains useful for operational forecasting because weekly behavior (weekday vs weekend profile) is not fully represented by a pure daily differencing operator and can still add predictive signal in supervised pipelines.


= Forecasting Model Comparison

This section presents the comparative forecasting experiment for hourly electricity demand under a realistic rolling evaluation setup. The goal was to assess statistical, machine learning, deep learning, and hybrid strategies under the same horizon and metrics, and then identify the most reliable approach for operational use.

== Experimental Design

The comparison was executed with a strict causal protocol to avoid leakage and to ensure that every model was evaluated under the same constraints.

- Rolling block horizon: H_roll = 24 hours.
- Validation strategy: expanding-window time split with a final holdout test period.
- Core metrics: $R^2$, MAE, RMSE, and MASE.
- Reference benchmark: the operational forecast available in the dataset.

== Feature Construction and Model Inputs

Feature engineering was designed to represent short-term dynamics, daily and weekly seasonality, and meteorological effects without introducing leakage. The target series was first cleaned using the operational forecast as the primary imputation source, and any remaining gaps were completed through time interpolation. Weather information from multiple cities was aggregated into a single hourly signal and aligned with the energy index to create a consistent exogenous block.

The final feature space combined calendar variables, weather covariates, autoregressive lags, and rolling statistics. Calendar information included hour, day of week, month, and a weekend indicator. Weather-related inputs included temperature (also converted to Celsius), humidity, wind speed, pressure, cloud coverage, and rain intensity. To capture persistence and recurrent consumption cycles, lag features at 1, 24, 48, and 168 hours were added. In parallel, rolling mean and rolling standard deviation were computed over 24-hour and 168-hour windows using shifted values to preserve strict causality.

In the modeling stage, these variables were consumed differently by each family. Baseline_ML used the complete engineered feature matrix. SARIMAX used a reduced exogenous subset focused on physically interpretable drivers (temperature, humidity, wind speed, wind generation, hour, and day of week), while SARIMA operated only on the univariate demand history. For the hybrid models, engineered features were also used to learn residual corrections: XGBoost received the feature matrix plus the SARIMA prediction, and the BaselineML_TimesFM_2_5 workflow modeled the residual trajectory produced by the baseline regressor.

#table(
	columns: 3,
	inset: 6pt,
	align: left,
	[*Feature Group*], [*Variables*], [*Used By*],

	[Calendar], [hour, dayofweek, month, is_weekend], [Baseline_ML, SARIMAX, Hybrid_SARIMA_XGB],
	[Weather], [temp_c, humidity, wind_speed, pressure, clouds_all, rain_1h], [Baseline_ML, SARIMAX, Hybrid_SARIMA_XGB],
	[Lags], [lag_1, lag_24, lag_48, lag_168], [Baseline_ML, Hybrid_SARIMA_XGB],
	[Rolling stats], [roll_mean_24, roll_std_24, roll_mean_168, roll_std_168], [Baseline_ML, Hybrid_SARIMA_XGB],
	[SARIMAX exogenous subset], [temp_c, humidity, wind_speed, generation wind onshore, hour, dayofweek], [SARIMAX],
	[Residual targets], [y - prediction residual series], [Hybrid_SARIMA_XGB, BaselineML_TimesFM_2_5]
)

== Models Evaluated

The set of evaluated models covered a broad spectrum of forecasting strategies. It included Baseline_Operative, which corresponds to the day-ahead operational forecast available in the dataset, and Baseline_ML, a feature-based linear machine learning baseline used as a strong reference among custom approaches. It also included SARIMA as the core seasonal statistical model and SARIMAX as its extension with exogenous covariates. To test hybrid statistical-learning behavior, the study incorporated Hybrid_SARIMA_XGB, where SARIMA predictions are corrected through XGBoost residual modeling. On the deep learning side, TimesFM_2_5 was evaluated as a standalone foundation model, and BaselineML_TimesFM_2_5 was tested as a residual hybrid in which TimesFM forecasts the error component of the baseline ML model.

Although each model belongs to a different methodological family, all of them were trained and evaluated under the same temporal protocol. This makes the comparison interpretable from both a scientific and operational perspective, because differences in performance can be attributed mainly to model behavior rather than to inconsistent data handling.



== SARIMA and SARIMAX

SARIMA and SARIMAX were prioritized because the demand series exhibits clear and persistent seasonality, and because the project contains a rich set of exogenous variables (especially weather and generation-related signals). AutoARIMA was initially considered, but it was discarded due to the high runtime per iteration and the large search space under this data scale. To keep the process tractable and reproducible, model orders were selected using ACF and PACF diagnostics instead of exhaustive automatic search.

== Consolidated Test Results

#table(
	columns: 6,
	inset: 6pt,
	align: left,
	[*Model*], [*$R^2$*], [*MASE*], [*MAE*], [*RMSE*], [*Relative Position*],

	[Baseline_Operative], [0.9928], [0.1241], [304.16], [415.12], [Best overall],
	[BaselineML_TimesFM_2_5], [0.9735], [0.2338], [573.12], [796.00], [Best custom model],
	[Baseline_ML], [0.9347], [0.3973], [974.07], [1249.95], [Strong ML baseline],
	[Hybrid_SARIMA_XGB], [0.9109], [0.4845], [1187.86], [1459.80], [Improves over pure SARIMA/SARIMAX],
	[TimesFM_2_5], [0.7045], [0.5925], [1452.75], [2658.40], [Weak as standalone],
	[SARIMA], [0.6045], [0.9714], [2381.55], [3075.44], [Lower accuracy],
	[SARIMAX], [0.6002], [0.9788], [2399.73], [3092.15], [Lower accuracy]
)

== Runtime Perspective

#table(
	columns: 3,
	inset: 6pt,
	align: left,
	[*Model*], [*Approx. Runtime (s)*], [*Comment*],

	[BaselineML_TimesFM_2_5], [5.24], [Fastest hybrid with strong accuracy],
	[TimesFM_2_5], [56.32], [Moderate runtime, limited standalone quality],
	[SARIMA], [454.39], [Slow iterative statistical fitting],
	[SARIMAX], [644.02], [Even slower with exogenous structure],
	[Hybrid_SARIMA_XGB], [859.13], [Highest runtime in this benchmark]
)

== Interpretation and Conclusions

The results support three main conclusions. First, in this specific experiment, classical statistical models did not convert the available seasonality and exogenous information into competitive final accuracy when compared with stronger ML and hybrid alternatives. Second, TimesFM alone did not perform well enough for direct deployment, confirming that foundation models can require adaptation when transferred to a domain-specific electrical load signal. Third, the residual hybrid strategy proved highly effective: when TimesFM is used to model the residual component of a robust baseline ML forecast, overall performance improves substantially and remains close to the strongest benchmark.

From an operational perspective, the most practical path is to keep a robust feature-based ML predictor as the core model and apply deep residual correction on top of it. This architecture achieved the best balance between accuracy and execution cost among the custom approaches, and it provides a scalable framework for future upgrades such as horizon-specific blending, residual feature enrichment, and scheduled retraining under drift conditions.


