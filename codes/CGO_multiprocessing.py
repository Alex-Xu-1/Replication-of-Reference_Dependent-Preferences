'''
    Note that this calculation is very time-consuming according to the logic of the paper.
    After implementing multiprocessing, it still takes about 60 min to finish the calculation.
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import njit
import time

from joblib import Parallel, delayed
from multiprocessing import Manager
from threading import Thread

df3 = pd.read_csv('../preprocessed_data/whole_df3.csv')
df4 = pd.read_csv('../preprocessed_data/whole_df4.csv')

# df3 = df3[df3['asset'].isin(range(1, 1500))]
# df3 = df3.sort_values(['asset', 'weekly_closing_date'])
# df3 = df3[(df3['weekly_closing_date'] >= '2000-01-01') & (df3['weekly_closing_date'] <= '2005-12-01')]

window_size = 260
min_weeks = 100


@njit
def compute_turnover_product(turnover_values):
    """
    Computes the cumulative product term for turnover values.

    :param turnover_values: Array of turnover values for the rolling window
    :return: Product of (1 - turnover) values for the given period
    """
    turnover_product = 1.0
    for turnover in turnover_values:
        turnover_product *= (1 - turnover)
    return turnover_product


def compute_rp_and_cgo_for_stock(stock_data, max_window=260):
    """
    Computes RP and CGO for a single stock.

    :param stock_data: DataFrame containing columns [week, weekly_close, weekly_turnover]
    :param max_window: Maximum rolling window length, default is 260 (5-year weekly data)
    :return: DataFrame with additional columns for RP and CGO
    """
    num_rows = len(stock_data)
    rp_values = np.full(num_rows, np.nan)
    cgo_values = np.full(num_rows, np.nan)

    for i in range(min_weeks, num_rows):
        rolling_window = min(i, max_window)
        weighted_sum = 0.0
        total_weight = 0.0

        for n in range(1, rolling_window + 1):
            weight = stock_data['weekly_turnover'].iloc[i - n]
            turnover_values = stock_data['weekly_turnover'].iloc[i - n + 1:i].values
            turnover_product = compute_turnover_product(turnover_values)

            weighted_sum += weight * turnover_product * stock_data['weekly_close'].iloc[i - n]
            total_weight += weight * turnover_product

        rp_value = weighted_sum / max(total_weight, 1e-5)  # Normalizing the weighted sum
        p_t_minus_1 = stock_data['weekly_close'].iloc[i - 1]
        cgo_value = (p_t_minus_1 - rp_value) / p_t_minus_1 if p_t_minus_1 != 0 else np.nan

        rp_values[i] = rp_value
        cgo_values[i] = cgo_value

    stock_data['RP'] = rp_values
    stock_data['CGO'] = cgo_values

    return stock_data


def wrapper_compute_rp_and_cgo(stock_data, progress_counter, progress_lock,  max_window):
    """
    Wrapper function to compute RP and CGO for a single stock and update the progress counter.

    :param stock_data: DataFrame containing stock data
    :param progress_counter: Shared counter to track progress
    :param max_window: Maximum rolling window length
    """
    result = compute_rp_and_cgo_for_stock(stock_data, max_window)
    ####### DEBUG LOGGER
    # print(f"Completed processing stock {stock_data['asset'].iloc[0]}")
    # Update the progress counter atomically
    with progress_lock:  # Ensure counter updates are atomic
        progress_counter.value += 1
    return result


def update_progress_bar(progress_counter, total_groups, pbar, start_time):
    """
    Updates the progress bar in the main process based on the shared counter.
    """
    while True:
        elapsed_time = time.time() - start_time
        avg_it_per_sec = max(progress_counter.value / elapsed_time, 1e-5)
        remaining_iters = total_groups - progress_counter.value
        eta = remaining_iters / avg_it_per_sec if avg_it_per_sec > 0 else float('inf')

        if np.isfinite(eta):
            eta_formatted = f"{int(eta // 60):02}:{int(eta % 60):02}" if eta < 3600 else \
                f"{int(eta // 3600):02}:{int((eta % 3600) // 60):02}:{int(eta % 60):02}"
        else:
            eta_formatted = "N/A"

        pbar.set_postfix(avg_it_s=f"{avg_it_per_sec:.2f}", eta=eta_formatted)
        pbar.update(progress_counter.value - pbar.n)

        if progress_counter.value >= total_groups:
            pbar.n = total_groups
            pbar.refresh()
            break

        time.sleep(0.1)  

    pbar.n = total_groups
    pbar.refresh()


def compute_rp_and_cgo(data, max_window=260, n_jobs=-1):
    """
    Calculates RP (return predictability) and CGO (conditional growth opportunities) for each stock.
    Uses multiprocessing to parallelize computation.

    :param data: DataFrame containing columns [asset, week, weekly_close, weekly_turnover]
    :param max_window: Maximum rolling window length, default is 260 (5-year weekly data)
    :param n_jobs: Number of CPU cores to use (default: -1 for all cores)
    :return: DataFrame with additional columns for RP and CGO
    """
    # Split data by asset
    grouped = sorted([group for _, group in data.groupby('asset')], key=len, reverse=True)
    for group in grouped:
        print(f"Asset {group['asset'].iloc[0]} has {len(group)} rows")
    total_groups = len(grouped)
    print(f"Processing {total_groups} asset groups...")

    # Use a multiprocessing Manager to track progress
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()

    # Initialize tqdm progress bar
    start_time = time.time()
    with tqdm(total=total_groups, desc="Processing Assets") as pbar:
        # Start a background thread to update the progress bar
        progress_thread = Thread(target=update_progress_bar, args=(progress_counter, total_groups, pbar, start_time))
        progress_thread.start()

        # Use joblib to parallelize the computation for each asset
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size='auto')(
            delayed(wrapper_compute_rp_and_cgo)(group, progress_counter, progress_lock, max_window) for group in grouped
        )

        # Ensure the progress bar finishes at 100%
        pbar.n = total_groups
        pbar.refresh()
        # print(f"Progress counter value: {progress_counter.value}/{total_groups}")
        progress_thread.join()

    # Concatenate all the processed groups back into a single DataFrame
    result_data = pd.concat(results, ignore_index=True)

    return result_data


if __name__ == "__main__":
    # data = df3
    data = df4
    
    result_df = compute_rp_and_cgo(data, n_jobs=12)
    # result_df.to_csv('../preprocessed_data/result_df_more_than_5_years_stocks.csv', index=False)
    
    result_df.to_csv('../preprocessed_data/result_df_more_than_10_years_stocks.csv', index=False)
    print(result_df)
