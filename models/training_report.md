# Kilter Board Grade Prediction - Model Training Report

*Generated on 2025-04-23 21:33*

## Model Comparison

| Metric | Neural Network | Random Forest |
|--------|---------------|---------------|
| Mean Absolute Error | 1.8346 | 1.8598 |
| Root Mean Squared Error | 2.4950 | 2.5150 |
| R² Score | 0.1900 | 0.1770 |
| Accuracy within 1 grade | 36.56% | 36.08% |
| Accuracy within 2 grades | 65.29% | 63.96% |
| Over-prediction rate | 53.83% | 54.41% |
| Under-prediction rate | 46.17% | 45.59% |
| Exact prediction rate | 0.00% | 0.00% |

**Best performing model: Neural Network** (based on MAE)

## Prediction Bias Analysis

- The Neural Network tends to overestimate grades (53.8% over vs 46.2% under).
- The Random Forest tends to overestimate grades (54.4% over vs 45.6% under).

## Random Forest Model

### Feature Importance

The Random Forest model identified the following features as most important for grade prediction:

| Feature | Importance |
|---------|------------|
| avg_distance | 0.0832 |
| density_left | 0.0812 |
| std_distance | 0.0791 |
| max_distance | 0.0729 |
| num_holds | 0.0537 |
| min_distance | 0.0380 |
| num_finish_holds | 0.0378 |
| density_top | 0.0316 |
| density_right | 0.0261 |
| is_listed | 0.0240 |

## Error Analysis

### Distribution of Prediction Errors

| Error Range | Neural Network | Random Forest |
|-------------|---------------|---------------|
| < 0.5 grade | 18.81% | 19.07% |
| < 1.0 grade | 36.56% | 36.08% |
| < 1.5 grades | 51.37% | 51.72% |
| < 2.0 grades | 65.29% | 63.96% |
| >= 2.0 grades | 34.71% | 36.04% |

## Conclusion

The Neural Network model achieved the best performance with a mean absolute error of 1.8346 grade points. Both models were able to predict grades within 1 grade of the actual value for about 36.1% of test routes and within 2 grades for about 64.0% of test routes.

Both models showed a tendency to overestimate grades, suggesting that there may be subtle factors affecting difficulty perception that aren't fully captured by the features.

### Key Findings

1. **avg_distance** is the most important feature for grade prediction.
2. The models can predict climbing grades with moderate accuracy (within 1 grade 36.1% of the time).
3. Accuracy increases to 64.0% when allowing predictions within 2 grades of the actual value.
4. Both models tend to overestimate route difficulty, with the Neural Network showing less bias.
5. Random Forest provides interpretable results through feature importance analysis.
