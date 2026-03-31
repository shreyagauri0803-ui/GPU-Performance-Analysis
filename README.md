GPU Kernel Performance AnalysisThis project implements Linear Regression and Logistic Regression from scratch using Batch Gradient Descent to analyze and predict the performance of SGEMM GPU kernels.

*Overview*
The goal of this project is to predict the computation time of GPU kernels based on various configuration parameters. Instead of using high-level libraries like Scikit-Learn for the model logic, the optimization algorithms were implemented manually to demonstrate a deep understanding of Gradient Descent mechanics.

*Dataset*
Source: UCI Machine Learning Repository - SGEMM GPU kernel performanceFeatures: 14 independent variables (MWG, NWG, KWG, VWM, VWN, etc.)Target: GPU computation time (Log-transformed for better regression scaling).

*Key Features*
Manual Implementation: Custom Gradient Descent functions for both Regression and Classification.Data Preprocessing: - Averaging multiple trial runs for a stable target variable.Outlier detection and removal using the IQR (Interquartile Range) method.Feature scaling via Z-score Normalization.Experimentation: - Comparison of different learning rates ($\alpha$) and convergence thresholds.Feature selection analysis (Comparing all features vs. random vs. highly correlated features).
 
*Tech Stack*
Language: Python 3.xLibraries: - Pandas & NumPy (Data manipulation)Matplotlib & Seaborn (Visualizing cost convergence and correlations)Scikit-Learn (Used only for data splitting and initial evaluation metrics)

*Summary of Results*
Linear Regression: Successfully minimized RMSE by tuning the learning rate and using Z-score normalization.
Logistic Regression: Achieved high classification accuracy by identifying that feature relevance (correlation) is more critical than the total number of features used.
Conclusion: The model performance confirmed that selecting the 8 most correlated features yields results comparable to using the full set of 14 features, proving that feature quality outweighs quantity.
