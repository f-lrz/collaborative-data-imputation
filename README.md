[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14187972.svg)](https://doi.org/10.5281/zenodo.14187972)
![Status](https://img.shields.io/badge/status-development-orange)

## Collaborative-Data-Imputation

<img src="img/colab_data_imputation.png" alt="Image Alt Text" width="700"/>

### Wind Power Data Reconstruction

In power system operations and electricity markets, missing data is a pervasive challenge in practice. Missing observations can arise from sensor faults, communication failures, or maintenance outages. This issue becomes particularly critical when large-scale, data-driven approaches are applied to point and probabilistic wind power forecasting, where data quality directly affects model performance and therefore decision making.

To address this, data imputation techniques—such as k-nearest neighbors (k-NN) and factor models—are commonly employed to reconstruct incomplete datasets before training forecasting models. Effective imputation ensures data completeness and consistency, which are essential for the reliability and accuracy of modern machine-learning–based forecasting methods.

### MLflow Experiments

MLflow is used to systematically compare and evaluate missing-data imputation algorithms, making it easier to identify the best-performing approach for a given dataset.

1. **Install UV (Dependency Manager)**

    ```bash
    pip install uv
    ```

2. **Install Project Dependencies**

    Install all required dependencies, including MLflow::

    ```bash
    uv sync
    ```

3. **Start Mlflow server**

    Launch the MLflow UI locally:

    ```bash
    uv run mlflow ui
    ```

3. **Run the Experiments**

    Execute the experiment pipeline:

    ```bash
    uv run main.py
    ```

4. **View Experiment Results**:

    Open your browser and navigato to `http://127.0.0.1:5000`.
     
    From the MLflow UI, you can explore:
    * Experiment runs
    * Model parameters and hyperparameters
    * Evaluation metrics
    * Logged artifacts (e.g., reconstructed datasets and plots)