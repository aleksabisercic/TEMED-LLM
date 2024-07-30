import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.tree import plot_tree

# Set seaborn theme and style for high-quality plots
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, label, dataset_name, model_name):
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    ds_name = dataset_name.split(".")[0]
    plt.savefig("graphs/{}_AUC_results_ML.png".format(ds_name), dpi=300, bbox_inches="tight")
    # save auc data for plotting
    df = pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "label": label, dataset_name: dataset_name}
    )
    df.to_csv("results/AUC_PLOT/{}_AUC_PLOT_DATA_{}.csv".format(ds_name, model_name), index=False)


# Function to plot precision, recall, and F1 score
def plot_scores(model_names, scores, dataset_name):
    df_scores = pd.DataFrame(
        scores, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    )
    ds_name = dataset_name.split(".")[0]
    df_scores.to_csv("results/{}_classification_results_ML.csv".format(ds_name), index=False)
    df_scores_melt = df_scores.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_scores_melt)
    plt.ylim([0.0, 1.0])
    plt.title("Model Performance Metrics")
    plt.legend(loc="lower right")
    print(df_scores_melt)
    # save plot
    plt.savefig(
        "graphs/{}_classification_results_ML.png".format(ds_name), dpi=300, bbox_inches="tight"
    )


def plot_multiple_auc(dataset_name):
    import glob

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")

    csv_files = glob.glob(f"{dataset_name}*.csv")
    for file in csv_files:
        df = pd.read_csv(file)
        label = df["label"][0]
        fpr = df["fpr"]
        tpr = df["tpr"]
        roc_auc = df["roc_auc"][0]

        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.legend(loc="lower right")
    ds_name = dataset_name.split(".")[0]
    plt.savefig(f"graphs/{ds_name}_combined_AUC_results_ML.png", dpi=300, bbox_inches="tight")
    plt.show()


# FEATURE IMPORTANCE PLOTS
def plot_decision_tree(
    decision_tree_pipe,
    all_feature_names,
    dataset_name,
    title,
    class_names=["Disease", "No Disease"],
    figsize=(48, 32),
):
    sns.set_theme(style="whitegrid")
    matplotlib.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        decision_tree_pipe.named_steps["classifier"],
        feature_names=all_feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=11,
        impurity=False,
        proportion=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=20)
    plt.xlabel("Decision Nodes", fontsize=18)
    plt.ylabel("Tree Depth", fontsize=18)

    sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)

    plt.show()
    fig.savefig(
        f"results/feature_importance/{dataset_name}_decision_tree_plot.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_shap_values(xgb_pipe, X_preprocessed, all_feature_names, dataset_name, title):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(xgb_pipe)
    shap_values = explainer(X_preprocessed)

    # Plot SHAP values
    shap.summary_plot(
        shap_values,
        features=X_preprocessed,
        feature_names=all_feature_names,
        title=title,
        show=False,
    )

    # Save the SHAP summary plot to a file
    plt.savefig(
        f"results/feature_importance/{dataset_name}_xgb_shap_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def xgboost_shap_plots(xgb_model, preprocessor, X, all_feature_names, dataset_name):
    # Initialize SHAP
    shap.initjs()

    # Using a random sample of the dataframe for better time computation
    X_sampled = X
    X_preprocessed = preprocessor.transform(X_sampled)

    # Explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_preprocessed)

    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_preprocessed, feature_names=all_feature_names, show=False)
    plt.savefig(
        f"results/feature_importance/{dataset_name}_XGBoost_shap_summary_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # SHAP Bar Plot
    shap.summary_plot(
        shap_values, X_preprocessed, feature_names=all_feature_names, plot_type="bar", show=False
    )
    plt.savefig(
        f"results/feature_importance/{dataset_name}_XGBoost_shap_bar_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Individual Force Plots
    for i in range(X_sampled.shape[0]):
        shap.force_plot(
            explainer.expected_value,
            shap_values[i, :],
            X_sampled.iloc[i, :],
            show=False,
            matplotlib=True,
        )
        plt.savefig(
            f"results/feature_importance/{dataset_name}_XGBoost_shap_force_plot_Example_{i}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        break
