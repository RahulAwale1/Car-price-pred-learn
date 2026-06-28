from pathlib import Path
import joblib
import pandas as pd
from src.config import BEST_MODEL_PATH
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def get_features_name(model_pipeline: Pipeline):
    preprocessor = model_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    return feature_names


def get_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]

    if not hasattr (model, "feature_importances_"):
        raise ValueError(
            "This model does not have feature_importances_. "
            "Use a tree-based model like RandomForestRegressor or GradientBoostingRegressor."
        )
    
    feature_names = get_features_name(model_pipeline)

    importance_values = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature_names": feature_names,
        "importance": importance_values,
    })

    importance_df = importance_df.sort_values(
        by = "importance",
        ascending = False,
    )

    return importance_df


def main():
    model = joblib.load(BEST_MODEL_PATH)

    importance_df = get_feature_importance(model)

    reports_path = Path("reports")
    reports_path.mkdir(parents=True, exist_ok=True)

    output_file = reports_path / "feature_importance.csv"
    importance_df.to_csv(output_file, index=False)
    
    top_features = importance_df.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature_names"], top_features["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_file = reports_path / "feature_importance.png"
    plt.savefig(plot_file)
    print("Feature Importance")
    print("------------------")
    print(importance_df)
    print(f"\nSaved feature importance to: {output_file}")
    print(f"Saved feature importance plot to: {plot_file}")


if __name__ == "__main__":
    main()