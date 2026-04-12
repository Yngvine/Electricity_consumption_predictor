== Forecasting Model Comparison

This section presents the comparative forecasting experiment for hourly electricity demand under a realistic rolling evaluation setup. The goal was to assess statistical, machine learning, deep learning, and hybrid strategies under the same horizon and metrics, and then identify the most reliable approach for operational use.

=== Experimental Design

The comparison was executed with a strict causal protocol to avoid leakage and to ensure that every model was evaluated under the same constraints.

- Rolling block horizon: H_roll = 24 hours.
- Validation strategy: expanding-window time split with a final holdout test period.
- Core metrics: $R^2$, MAE, RMSE, and MASE.
- Reference benchmark: the operational forecast available in the dataset.

=== Models Evaluated

The set of evaluated models covered a broad spectrum of forecasting strategies. It included Baseline_Operative, which corresponds to the day-ahead operational forecast available in the dataset, and Baseline_ML, a feature-based linear machine learning baseline used as a strong reference among custom approaches. It also included SARIMA as the core seasonal statistical model and SARIMAX as its extension with exogenous covariates. To test hybrid statistical-learning behavior, the study incorporated Hybrid_SARIMA_XGB, where SARIMA predictions are corrected through XGBoost residual modeling. On the deep learning side, TimesFM_2_5 was evaluated as a standalone foundation model, and BaselineML_TimesFM_2_5 was tested as a residual hybrid in which TimesFM forecasts the error component of the baseline ML model.

Although each model belongs to a different methodological family, all of them were trained and evaluated under the same temporal protocol. This makes the comparison interpretable from both a scientific and operational perspective, because differences in performance can be attributed mainly to model behavior rather than to inconsistent data handling.



=== SARIMA and SARIMAX

SARIMA and SARIMAX were prioritized because the demand series exhibits clear and persistent seasonality, and because the project contains a rich set of exogenous variables (especially weather and generation-related signals). AutoARIMA was initially considered, but it was discarded due to the high runtime per iteration and the large search space under this data scale. To keep the process tractable and reproducible, model orders were selected using ACF and PACF diagnostics instead of exhaustive automatic search.

=== Consolidated Test Results

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

=== Runtime Perspective

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

=== Interpretation and Conclusions

The results support three main conclusions. First, in this specific experiment, classical statistical models did not convert the available seasonality and exogenous information into competitive final accuracy when compared with stronger ML and hybrid alternatives. Second, TimesFM alone did not perform well enough for direct deployment, confirming that foundation models can require adaptation when transferred to a domain-specific electrical load signal. Third, the residual hybrid strategy proved highly effective: when TimesFM is used to model the residual component of a robust baseline ML forecast, overall performance improves substantially and remains close to the strongest benchmark.

From an operational perspective, the most practical path is to keep a robust feature-based ML predictor as the core model and apply deep residual correction on top of it. This architecture achieved the best balance between accuracy and execution cost among the custom approaches, and it provides a scalable framework for future upgrades such as horizon-specific blending, residual feature enrichment, and scheduled retraining under drift conditions.
