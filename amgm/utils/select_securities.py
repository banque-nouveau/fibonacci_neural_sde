import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import sys
from pathlib import Path

import amgm.config as amgm_config



def find_least_correlated_securities(security_data: pd.DataFrame, start_date_str: str, end_date_str: str, n_select: int = 500):
    """
    Finds the n_select least correlated securities (IssueIds) from the given DataFrame
    within a specified date range, ensuring full year data coverage for chosen securities.

    The correlation is determined by calculating the average absolute Pearson correlation
    of each security's price movements against all other securities in the filtered set.
    IssueIds with the lowest average absolute correlation are considered the least correlated.

    Args:
        security_data (pd.DataFrame): DataFrame containing security prices with columns
                                      "IssueId", "Date_prices", and a price column (e.g., "Price").
        start_date_str (str): The start date string in 'YYYY-MM-DD' format (e.g., "2015-01-01").
        end_date_str (str): The end date string in 'YYYY-MM-DD' format (e.g., "2020-12-31").
        n_select (int): The number of least correlated IssueIds to select. Defaults to 500.

    Returns:
        list: A list of IssueIds (strings) that are among the n_select least correlated.
              Returns an empty list if no suitable securities are found.
    """
    # Convert date strings to datetime objects for accurate filtering
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    print(f"\n--- Starting correlation analysis for dates {start_date_str} to {end_date_str} ---")

    # Step 1: Filter data by the specified date range
    # Ensure 'Date_prices' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(security_data["Date_prices"]):
        print("Converting 'Date_prices' to datetime format...")
        security_data["Date_prices"] = pd.to_datetime(security_data["Date_prices"])

    df_filtered_dates = security_data[
        (security_data["Date_prices"] >= start_date) &
        (security_data["Date_prices"] <= end_date)
    ].copy() # Use .copy() to prevent SettingWithCopyWarning

    print(f"1. Data filtered to date range. Total rows in range: {len(df_filtered_dates)}")

    # Determine all full years expected within the date range
    all_years_in_range = set(range(start_date.year, end_date.year + 1))
    print(f"   Expected years for full data coverage: {sorted(list(all_years_in_range))}")

    # Step 2: Identify IssueIds that have data for ALL years in the specified range
    issues_with_complete_data = []
    print("2. Checking for complete yearly data coverage for each IssueId...")

    # Group by 'IssueId' and check if all required years are present for each group
    # This loop is optimized by using `groupby` which is C-optimized under the hood for Pandas.
    for issue_id, group in df_filtered_dates.groupby("IssueId"):
        unique_years_for_issue = set(group["Date_prices"].dt.year.unique())
        # Check if the set of all required years is a subset of years present for the issue
        if all_years_in_range.issubset(unique_years_for_issue):
            issues_with_complete_data.append(issue_id)

    print(f"   Found {len(issues_with_complete_data)} IssueIds with complete data coverage across all years.")

    if not issues_with_complete_data:
        print("   No IssueIds found with complete data for the specified date range. Returning empty list.")
        return []

    # Filter the DataFrame to include only the IssueIds with complete data
    df_complete_issues = df_filtered_dates[
        df_filtered_dates["IssueId"].isin(issues_with_complete_data)
    ].copy()

    # Step 3: Pivot the data to prepare for correlation calculation
    # Reshape the DataFrame: 'Date_prices' as index, 'IssueId' as columns, 'Price' as values.
    # This creates a time series DataFrame where each column is an IssueId's price history.
    print("3. Pivoting data for correlation matrix calculation...")
    try:
        price_df = df_complete_issues.pivot(index="Date_prices", columns="IssueId", values="ClAdjUsd")
        print(f"   Price DataFrame shape after pivoting: {price_df.shape}")

    except ValueError as e:
        # Handle cases where there might be duplicate date entries for a single IssueId
        print(f"   Warning: Error pivoting data ({e}). This might happen if duplicate dates exist for an IssueId.")
        print("   Attempting to resolve by averaging prices for duplicate dates per IssueId...")
        df_complete_issues = df_complete_issues.groupby(["IssueId", "Date_prices"])["ClAdjUsd"].mean().reset_index()
        price_df = df_complete_issues.pivot(index="Date_prices", columns="IssueId", values="ClAdjUsd")
        print(f"   Price DataFrame shape after pivoting: {price_df.shape}")

    except Exception as e:
        print(f"   An unexpected error occurred during pivoting: {e}. Returning empty list.")
        return []

    initial_cols = price_df.shape[1]
    missing_threshold = 0.5
    # Calculate the proportion of missing values for each column
    missing_proportions = price_df.isnull().sum() / len(price_df)
    # Identify columns to drop based on the threshold
    cols_to_drop = missing_proportions[missing_proportions > missing_threshold].index
    if not cols_to_drop.empty:
        price_df.drop(columns=cols_to_drop, inplace=True)
        print(f"   Dropped {len(cols_to_drop)} columns with more than {missing_threshold*100:.0f}% missing values.")
    else:
        print(f"   No columns dropped based on missing value threshold ({missing_threshold*100:.0f}%).")
    
    print("   Filling any NaN values in pivoted data using forward fill...")
    price_df.ffill(inplace=True) # Forward fill any NaNs
    
    # Drop any columns (IssueIds) that might still contain NaN values after pivoting.
    # This ensures that only securities with a full time series for all dates are considered.
    price_df.dropna(axis=1, inplace=True)
    print(f"   Price DataFrame shape after pivoting and dropping NaNs: {price_df.shape}")

    # Check if there's enough data for correlation calculation
    if price_df.empty or price_df.shape[1] < 2:
        print("   Not enough complete securities or data points to calculate meaningful correlations. Returning empty list.")
        return []

    # Step 4: Calculate the Pearson correlation matrix
    print("4. Calculating Pearson correlation matrix...")
    correlation_matrix = price_df.corr(method='pearson')
    print(f"   Correlation matrix calculated. Shape: {correlation_matrix.shape}")

    # Step 5: Find the n_select least correlated IssueIds
    # We define "least correlated" as having the lowest average absolute correlation
    # with all other securities in the set.
    print(f"5. Determining the {n_select} least correlated IssueIds...")

    average_abs_correlations = pd.Series(dtype=float)
    # Iterate through each column (IssueId) in the correlation matrix
    for col in correlation_matrix.columns:
        # Get absolute correlation values for the current column, excluding self-correlation (diagonal)
        # .drop(col) removes the correlation of an issue with itself (which is always 1).
        abs_corrs = correlation_matrix[col].abs().drop(col)
        if not abs_corrs.empty:
            # Calculate the mean of these absolute correlations
            average_abs_correlations.loc[col] = abs_corrs.mean()
        else:
            # This case indicates an IssueId with no other securities to correlate against
            average_abs_correlations.loc[col] = np.nan

    # Remove any IssueIds that resulted in NaN (e.g., if there was only one security remaining)
    average_abs_correlations.dropna(inplace=True)

    if average_abs_correlations.empty:
        print("   No securities with valid average absolute correlations found. Returning empty list.")
        return []

    # Sort the IssueIds based on their average absolute correlation in ascending order
    # The lowest values will be at the top, indicating least correlation.
    least_correlated_issues = average_abs_correlations.sort_values(ascending=True)

    # Select the top 'n_select' IssueIds from the sorted list
    selected_issue_ids = least_correlated_issues.head(n_select).index.tolist()

    print(f"--- Successfully selected {len(selected_issue_ids)} least correlated IssueIds. ---")
    return selected_issue_ids



def sample_securities_from_industries(
    security_data, 
    security_universe, 
    num_samples: int = 500, 
    start_date: str = "2015-01-01", 
    end_date: str = "2020-12-31"
):
    """
    Sample securities from industries in the security universe,
    ensuring that all sampled IssueIds have data for all years between start_date and end_date (inclusive).
    """
    # Convert start and end date to years
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    required_years = set(str(y) for y in range(start_year, end_year + 1))

    industry_counts = security_universe['SectorCode_security_universe'].value_counts()
    print("Number of securities from each sector:")
    for sector, count in industry_counts.items():
        print(f"Sector {sector}: {count} securities")
    samples_per_industry = (industry_counts / industry_counts.sum() * num_samples).round().astype(int)
    all_samples = []

    # Precompute which IssueIds have data for all required years
    # Assume security_data has a column "IssueId" and "Date_prices"
    issueid_years = (
        security_data
        .assign(year=security_data["Date_prices"].astype(str).str[:4])
        .groupby("IssueId")["year"]
        .agg(lambda years: set(years))
    )
    eligible_issueids = set(issueid for issueid, years in issueid_years.items() if required_years.issubset(years))

    for industry_code, n_samples in samples_per_industry.items():
        ids = security_universe[security_universe['SectorCode_security_universe'] == industry_code]['IssueId'].unique()
        # Only keep ids that are eligible
        eligible_ids = [iid for iid in ids if iid in eligible_issueids]
        if len(eligible_ids) < n_samples:
            print(f"Warning: Requested {n_samples} samples for industry {industry_code}, but only {len(eligible_ids)} eligible (with full year coverage) available.")
            n_samples = len(eligible_ids)
        if n_samples > 0:
            all_samples.extend(random.sample(list(eligible_ids), n_samples))
    # If rounding caused us to have more or less than num_samples, adjust:
    if len(all_samples) > num_samples:
        all_samples = random.sample(all_samples, num_samples)
    elif len(all_samples) < num_samples:
        # If less, sample more randomly from the full eligible set to reach num_samples
        remaining_ids = eligible_issueids - set(all_samples)
        n_needed = num_samples - len(all_samples)
        if n_needed > 0 and len(remaining_ids) > 0:
            n_to_add = min(n_needed, len(remaining_ids))
            all_samples.extend(random.sample(list(remaining_ids), n_to_add))
    return all_samples[:num_samples]


def preprocess_dataframes(security_data, security_universe):
    # Process security_data
    security_data['Date'] = pd.to_datetime(security_data['Date'])
    security_data['Date'] = security_data['Date'].dt.strftime('%Y-%m-%d')
    security_data.rename(columns={'Date': 'Date_prices'}, inplace=True)

        # Process security_universe
    security_universe['Date'] = pd.to_datetime(security_universe['Date'])
    security_universe['Date'] = security_universe['Date'].dt.strftime('%Y-%m-%d')
    security_universe.rename(columns={'Date': 'Date_security'}, inplace=True)
    security_universe.rename(columns={'SectorCode': 'SectorCode_security_universe'}, inplace=True)

    return security_data, security_universe


def get_security_data(data_folder):
    # Read all .txt files in the folder
    txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

    # Read the content of each file as a DataFrame
    dataframes = {}
    for file_name in txt_files:
        file_path = os.path.join(data_folder, file_name)
        try:
            df = pd.read_csv(file_path, sep='\t')  
            dataframes[file_name] = df
        except Exception as e:
            print(f"Could not read {file_name} as a table: {e}")

    return dataframes['security_data.txt'], dataframes['security_universe.txt']


########################################################################################################

if __name__ == '__main__':

    security_data, security_universe = get_security_data(data_folder=amgm_config.dataset_root / "data-20250505")
    
    # Apply preprocessing
    security_data, security_universe = preprocess_dataframes(security_data, security_universe)
    
    # Call the function to find the least correlated securities
    selected_least_correlated_issues = find_least_correlated_securities(
        security_data=security_data,
        start_date_str="2014-01-01",
        end_date_str="2020-12-31",
        n_select=500 # The number of least correlated IssueIds you want to retrieve
    )
    

    print("\nSelected Top 500 Least Correlated Issue IDs (first 10 shown):")
    print(selected_least_correlated_issues[:10])
    print(f"\nTotal selected: {len(selected_least_correlated_issues)} Issue IDs.")


    selected_securities_from_industries = sample_securities_from_industries(
        security_data=security_data,
        security_universe=security_universe,
        start_date="2014-01-01",
        end_date="2020-12-31",  
        num_samples=500)

    print(f"sampled {len(selected_securities_from_industries)} securities from industries")
    print(selected_securities_from_industries[:10])
    print(f"\nTotal selected: {len(selected_securities_from_industries)} Issue IDs.")

    # --- Code to write selected Issue IDs to a text file ---
    
    data_dir = amgm_config.dataset_root
    try:
        output_filename = "selected_issue_ids_least_correlated.txt"
        with open(data_dir / output_filename, 'w') as f:
            for issue_id in selected_least_correlated_issues:
                f.write(f"{issue_id}\n")
        print(f"\nSuccessfully wrote selected Issue IDs to {output_filename}")

        # --- Code to write selected Issue IDs to a text file ---
        output_filename = "selected_issue_ids_industries.txt"
        with open(data_dir / output_filename, 'w') as f:
            for issue_id in selected_securities_from_industries:
                f.write(f"{issue_id}\n")
        print(f"\nSuccessfully wrote selected Issue IDs to {output_filename}")
    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}")
