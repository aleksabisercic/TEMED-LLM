import pandas as pd

def count_none_values(json_obj):
    """
    Count the number of None values in the JSON object.
    :param json_obj: a json object.
    :return: int, count of None values in the json.
    """
    return sum(1 for value in json_obj.values() if value is None)

def count_value_in_medical_report(value, medical_report):
    """Count the number of occurrences of a specific value in the medical_report"""
    return 1 if str(value) in medical_report else 0

def compare_dicts(dict1, dict2, medical_report):
    """
    Compare two dictionaries and also count the None values and occurrences in the medical report.
    
    :param dict1: Ground truth  dictionary.
    :param dict2: Extracted dictionary to compare.
    :param medical_report: a medical report as a string.
    
    :return: comparison_result: a dictionary with comparison results.
    :return: none_count: int, count of None values.
    :return: medical_report_count: int, count of occurrences in the medical report.
    """
    comparison_result = {}
    none_count = 0
    medical_report_count = 0
    if type(dict2) != dict:
        return {key: False for key in dict1.keys()}, none_count, medical_report_count
    for key in dict1.keys():
        if dict1.get(key) == dict2.get(key):
            comparison_result[key] = True
        else:
            if dict2.get(key) is None:
                none_count += 1
                medical_report_count += count_value_in_medical_report(dict1.get(key), medical_report)
                comparison_result[key] = True if str(dict1.get(key)) not in medical_report else False
            if str(dict1.get(key)) not in medical_report and str(dict2.get(key)) in medical_report:
                comparison_result[key] = True                 
            else:
                comparison_result[key] = False
    return comparison_result, none_count, medical_report_count

def create_comparison_columns(df: pd.DataFrame, 
                              results_column_name="result_json", 
                              medical_report_column_name="medical_report", 
                              original_json_column_name="json_row"):
    """
    Create comparison columns in the given DataFrame.
    
    :param df: pandas DataFrame.
    :param results_column_name: name of the column with the extracted results.
    :param medical_report_column_name: name of the column with the medical report.
    :param original_json_column_name: name of the column with the original json.
    
    :return: pandas DataFrame with additional comparison columns.
    """
    comparisons = []
    none_counts = []
    medical_report_counts = []

    for _, row in df.iterrows():
        comparison, none_count, medical_report_count = compare_dicts(
            row[original_json_column_name], row[results_column_name], row[medical_report_column_name]
        )
        comparisons.append(comparison)
        none_counts.append(none_count)
        medical_report_counts.append(medical_report_count)

    df["validation_result"] = [all(comp.values()) for comp in comparisons]
    
    for key in df.iloc[0][original_json_column_name].keys():
        df[f"{key}_match"] = [comparison[key] for comparison in comparisons]

    df["none_count"] = none_counts
    df["false_none"] = medical_report_counts

    return df
    
def get_statistics(df: pd.DataFrame, dataset_name):
    total_rows = len(df)
    valid_rows = len(df[df["validation_result"] == True])
    invalid_rows = len(df[df["validation_result"] == False])
    none_rows = df["none_count"].sum()
    false_none_rows = df["false_none"].sum()
    cost_mean = df["result_cost"].mean()
    cost_mean_gpt4 = df["result_completion_tokens"].mean()*0.045/1000
    
    data = {
        "Total Rows": [total_rows],
        "Valid Rows": [valid_rows],
        "Invalid Rows": [invalid_rows],
        "None Rows": [none_rows],
        "False None Rows": [false_none_rows],
        "Cost Per Extraction (ChatGPT)": [cost_mean],
        "Cost Per Extraction (GPT4)": [cost_mean_gpt4],
    }

    stats_df = pd.DataFrame(data)
    ds_name = dataset_name.split(".")[0]
    stats_df.to_csv(f"results/extraction/{ds_name}_extraction_stats.csv")
    return stats_df