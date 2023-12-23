
# Optimizing an ML Pipeline in Azure

## Problem Statement:

Binary Classification Task:

Dataset: UCI Bank Marketing dataset (bankmarketing_train.csv)
Features: 21 numerical and categorical features
Target Variable: 'y' (binary, indicating term deposit subscription)
Objective: Develop a model to accurately predict term deposit subscription outcomes.
Modeling Approach:

Model 1: Logistic Regression with Hyperparameter Tuning

Leveraged Azure ML Python SDK and HyperDrive for hyperparameter optimization.
Achieved best accuracy of 91.10%.
Model 2: Azure AutoML with Automated Model Selection and Training

Explored diverse model architectures (LightGBM, XGBoostClassifier, RandomForest, VotingEnsemble, StackEnsemble).
Yielded best accuracy of 91.74%.

## Best Model:

Model Performance:

Top-Performing Model: Voting Ensemble incorporating XGBoost classifiers.
Feature Scaling: Standard Scaler applied for normalization.
Accuracy: 91.74%.
Key Technical Details:

Ensemble Learning: Combines multiple models for enhanced performance.
XGBoost: Powerful gradient boosting algorithm known for accuracy and efficiency.
Standard Scaler: Normalizes features by centering and scaling to a standard deviation of 1, often beneficial for gradient-based algorithms.

## Data Acquisition and Preprocessing:

Dataset Retrieval: Leveraged Azure Machine Learning's TabularDatasetFactory class to fetch the Bank Marketing dataset (bankmarketing_train.csv) from the specified URL.
Data Cleaning and Preprocessing: Employed the clean_data() function to perform essential preprocessing steps, including:
One-Hot Encoding: Transformed categorical features into numerical representations for compatibility with machine learning algorithms.
Train-Test Split: Partitioned the preprocessed dataset into training (70%) and testing (30%) sets for model development and evaluation, respectively.

## Model Selection and Training:

Algorithm: Logistic Regression, a widely used statistical classification method for predicting probabilities of binary outcomes.
Training Script: train.py, containing model training logic and hyperparameter tuning setup.
HyperDrive Integration:
Estimator: Configured for Logistic Regression training.
HyperDrive Configuration: Defines parameter sampling, early stopping policy, and run settings.
Azure ML Environment:
Workspace and Experiment: Established for model development and tracking.
Compute Cluster: "Standard_DV_V2" VMs (4 nodes) provisioned for computational resources.

## Hyperparameter Tuning Process:

Parameter Space Exploration:
Leveraged Random Sampling for efficient hyperparameter search, striking a balance between exploration and computational cost.
Early Termination of Underperforming Runs:
Implemented Bandit policy to adaptively allocate resources to promising configurations, maximizing overall efficiency.
HyperDrive Experimentation:
Launched HyperDrive runs with specified configurations to systematically explore the hyperparameter space and identify optimal values.

## Benefits Of The Parameter Sampler:

Hyperparameter Tuning: Applied to optimize the inverse regularization parameter (--C) and maximum iterations (--max_iter), both crucial for model performance and convergence.
Search Space Definition: Values for --C set to (0.1, 1) and --max_iter to (50, 100, 150, 200) to encompass a range of potential configurations.
Efficient Exploration: Random sampling enables efficient navigation of this hyperparameter space while supporting early termination of underperforming runs.

## Benefits Of The Early Stopping Policy:

Strategic Resource Allocation via Early Stopping:

Optimizing Efficiency with Early Termination:

Bandit Policy Implementation: Employed for adaptive resource allocation through the termination of underperforming runs, enhancing computational efficiency and model exploration.
Hyperparameter Tuning: slack_factor=0.1, evaluation_interval=1, and delay_evaluation=5 set to guide strategic termination decisions.
HyperDrive Experiment Setup:

SKLearn Estimator Configuration: Configured with train.py training script, directory path, and cluster name for model training.
HyperDriveConfig Establishment: Defined with key parameters:
Estimator: SKLearn-based
Parameter Sampling: Random Sampling for efficient hyperparameter exploration
Policy: Bandit for early stopping
Primary Metric: Accuracy (objective: maximization)
Maximum Runs: 10
Maximum Concurrent Runs: 4
Experiment Execution and Monitoring:

Experiment Submission: HyperDrive experiment initiated for systematic model training and evaluation.
Visualization: Experiment progress tracked using the RunDetails(hyperdrive_run).show() widget for real-time insights.
Best Model Identification: Upon completion, the model achieving the highest accuracy is retrieved and saved in the designated outputs folder.

---

# AutoML

Data Acquisition and Preparation:

Retrieved dataset (bankmarketing_train.csv) using TabularDatasetFactory class.
Preprocessed and cleaned data using clean_data() function.
Partitioned dataset into 70% training and 30% testing sets.
AutoML Configuration:

Defined AutoMLConfig with crucial parameters:
Experiment Timeout: 30 minutes
Task Type: Classification
Primary Metric: Accuracy
Label Column: 'Y'
Cross Validation Folds: 5
Enabled exploration of diverse model architectures (LightGBM, XGBoostClassifier, RandomForest, VotingEnsemble, StackEnsemble).
Incorporated various preprocessing techniques (Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler).
Experiment Execution and Monitoring:

Submitted AutoML experiment for automated model training and evaluation.
Tracked experiment progress using RunDetails(automl_run).show() widget.
Best Model Retrieval:

Upon completion, identified and saved the model with the highest accuracy in the outputs folder.

---

## Pipeline comparison

Key Observations from Pipeline Comparison:

Marginally Superior Performance with AutoML: AutoML achieved a slightly higher best accuracy of 91.74% compared to HyperDrive's 91.10%, suggesting a potential advantage in model selection and optimization.
Model Exploration and Architecture:

Diverse Model Space Exploration with AutoML: AutoML demonstrated flexibility by exploring a broader range of model architectures, including Voting Ensembles, which can often enhance performance by combining multiple predictions.
Single Algorithm Focus in HyperDrive: HyperDrive concentrated on hyperparameter tuning for a single algorithm (Logistic Regression), limiting its model exploration scope.
Workflow and Experiment Design:

AutoML's Streamlined Workflow: AutoML integrated data retrieval, cleaning, preprocessing, and model training within its automated process, streamlining development.
HyperDrive's Manual Preprocessing and Configuration: HyperDrive required manual preprocessing and configuration of the training script and HyperDriveConfig, suggesting a more hands-on approach.
Experiment Flexibility and Potential:

Extended Exploration with Longer Timeouts: AutoML's longer experiment timeout (30 minutes) enabled a wider range of model and preprocessing exploration, potentially leading to better outcomes.
HyperDrive's Focused Exploration: HyperDrive's experiment design favored focused tuning within a specific algorithm space, potentially missing out on broader model possibilities.
Conclusion:

AutoML's Advantages for Model Exploration: AutoML demonstrated advantages in terms of model exploration flexibility, ensemble techniques, and overall workflow efficiency.
HyperDrive's Strengths in Focused Tuning: HyperDrive remains valuable for targeted hyperparameter optimization within specific algorithm spaces.
Recommendations: Consider prioritizing AutoML for initial model exploration and potential superior performance, while HyperDrive can be valuable for fine-tuning specific algorithms.

---

## Key Areas for Future Research and Optimization:

Algorithmic Exploration and Hyperparameter Tuning:

Algorithm Diversification: Extend HyperDrive experiments to encompass a broader range of algorithms, potentially uncovering performance-enhancing alternatives.
Comprehensive Hyperparameter Search: Investigate Grid and Bayesian sampling strategies to complement Random Sampling, aiming for more thorough hyperparameter space exploration.
AutoML Exploration Duration:

Timeout Expansion: Increase AutoML's allocated runtime to enable evaluation of a wider spectrum of models, potentially leading to superior performance.
Class Imbalance Mitigation and Performance Enhancement:

Sampling Techniques: Address dataset class imbalance using SMOTE or ADASYN oversampling methods to improve model robustness and accuracy.
Hyperparameter and Cross-Validation Tuning: Experiment with alternative hyperparameter configurations and increase cross-validation folds for enhanced model generalization and accuracy.
Evaluation Metric Expansion:

AUC-Weighted Consideration: Incorporate AUC-weighted metric to complement accuracy, providing a more comprehensive assessment of model performance across classes, particularly in imbalanced scenarios.
