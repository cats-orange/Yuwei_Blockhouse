#!/usr/bin/env python
# coding: utf-8

# In[16]:


# backtest.py
# Yuwei's Blockhouse Trial Task Backtest
# Implements the Cont–Kukanov allocator, three baselines, parameter tuning, and reporting per the trial task specification.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json

# 1. Data Loading & Preparation

def load_and_prepare(filepath: str):
    """
    Load the Level-1 CSV feed, parse timestamps, dedupe, and pivot to one row per timestamp.
    Returns:
      df_pivot   : DataFrame indexed by ts_event with ask_px_00 and ask_sz_00 for each venue
      price_cols : list of price column names
      size_cols  : list of size column names
    """
    df = pd.read_csv('l1_day.csv')
    # parse and sort
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts_event')
    # dedupe: first quote per venue per timestamp
    df = df.drop_duplicates(subset=['ts_event','publisher_id'], keep='first')
    # pivot level-1 ask
    lvl1 = df[['ts_event','publisher_id','ask_px_00','ask_sz_00']]
    pivot = lvl1.pivot(index='ts_event', columns='publisher_id')
    pivot.columns = [f"{c[0]}_venue{c[1]}" for c in pivot.columns]
    # identify price vs size cols
    price_cols = [c for c in pivot.columns if c.startswith('ask_px_00')]
    size_cols  = [c for c in pivot.columns if c.startswith('ask_sz_00')]
    # fill missing
    pivot[price_cols] = pivot[price_cols].fillna(np.inf)
    pivot[size_cols]  = pivot[size_cols].fillna(0)
    return pivot, price_cols, size_cols

# 2. Cost Calculation

def compute_cost(split, prices, sizes, order_size,
                 lambda_over, lambda_under, theta_queue,
                 fees, rebates):
    """
    Compute total expected cost for a given split:
      - I didn’t write the for loop and used NumPy vector operations instead
      - fills at price+fee
      - maker rebates on unfilled shares
      - under/over execution penalties
      - queue-risk penalty
    """
    executed  = np.minimum(split, sizes)
    cash_spent = np.dot(executed, prices + fees)
    cash_spent -= np.dot(np.maximum(split - sizes, 0), rebates)
    underfill = max(order_size - executed.sum(), 0)
    overfill  = max(executed.sum() - order_size, 0)
    risk_pen  = theta_queue * (underfill + overfill)
    cost_pen  = lambda_under * underfill + lambda_over * overfill
    return cash_spent + risk_pen + cost_pen

# 3. Cont–Kukanov Allocator

def allocate(remaining, prices, sizes,
             lambda_over, lambda_under, theta_queue,
             fees, rebates, step=100):
    """
    Brute-force candidate splits in multiples of 'step' across venues,
    scoring each with compute_cost and returning the cheapest.
    """
    N = len(prices)
    # start with an empty allocation list
    candidates = [[]]
    for v in range(N):
        new_candidates = []
        for cand in candidates:
            used = sum(cand)
            max_share = remaining - used
            if v < N - 1:
                # try all possible k-step allocations up to remaining
                for k in range(0, max_share + 1, step):
                    new_candidates.append(cand + [k])
            else:
                # last venue gets the leftover
                new_candidates.append(cand + [max_share])
        candidates = new_candidates

    # evaluate each candidate
    best_cost, best_split = float('inf'), None
    for cand in candidates:
        arr = np.array(cand, dtype=int)
        cost = compute_cost(arr, prices, sizes, remaining,
                            lambda_over, lambda_under, theta_queue,
                            fees, rebates)
        if cost < best_cost:
            best_cost, best_split = cost, arr
    return best_split

# 4. Baseline Strategies

def run_best_ask(df, price_cols, size_cols, order_size, fees, rebates):
    """Hit the lowest ask each snapshot."""
    rem, cash = order_size, 0.0
    for _, row in df.iterrows():
        if rem <= 0: break
        prices = row[price_cols].values
        sizes  = row[size_cols].values
        i = np.argmin(prices)
        take = min(rem, sizes[i])
        cash += take * (prices[i] + fees[i])
        rem -= take
    filled = order_size - max(rem,0)
    return cash, cash/filled


def run_twap(df, price_cols, size_cols, order_size, fees, rebates, bucket_secs=60):
    """
    Conduct TWAP in fixed time buckets,
    splitting remaining equally and rounding up to 1 share per bucket.
    """
    df2 = df.copy()
    df2['bucket'] = (df2.index.astype(np.int64) // (bucket_secs*1e9)).astype(int)
    groups = df2.groupby('bucket')
    rem, cash = order_size, 0.0
    for _, bucket in groups:
        if rem <= 0: break
        share = max(int(np.ceil(rem / len(groups))), 1)
        # allocate share in this bucket across snapshots evenly
        for _, row in bucket.iterrows():
            if rem <= 0: break
            prices = row[price_cols].values; sizes = row[size_cols].values
            weights = sizes/sizes.sum() if sizes.sum()>0 else np.zeros_like(sizes)
            alloc = np.maximum((share * weights).astype(int), 1)
            alloc = np.minimum(alloc, rem)
            execd = np.minimum(alloc, sizes)
            cash += np.dot(execd, prices+fees) - np.dot(alloc-execd, rebates)
            rem -= execd.sum()
    filled = order_size - max(rem,0)
    return cash, cash/filled


def run_vwap(df, price_cols, size_cols, order_size, fees, rebates):
    """Split each snapshot by size weights."""
    rem, cash = order_size, 0.0
    for _, row in df.iterrows():
        if rem <= 0: break
        prices = row[price_cols].values; sizes = row[size_cols].values
        weights = sizes/sizes.sum() if sizes.sum()>0 else np.zeros_like(sizes)
        alloc = (rem * weights).astype(int)
        execd = np.minimum(alloc, sizes)
        cash += np.dot(execd, prices+fees) - np.dot(alloc-execd, rebates)
        rem -= execd.sum()
    filled = order_size - max(rem,0)
    return cash, cash/filled

# 5. Parameter Grid Search for Router

def grid_search(df, price_cols, size_cols, order_size=5000):
    lambdas_over  = [0.01, 0.05, 0.10, 0.20]
    lambdas_under = [0.01, 0.05, 0.10, 0.20]
    thetas        = [0.00, 0.05, 0.10, 0.20]
    fees    = np.zeros(len(price_cols))
    rebates = np.zeros(len(price_cols))
    best = {'cost': float('inf')}
    for lo, lu, th in itertools.product(lambdas_over, lambdas_under, thetas):
        rem, cash = order_size, 0.0
        for _, row in df.iterrows():
            if rem <= 0: break
            p = row[price_cols].values; s = row[size_cols].values
            split = allocate(rem, p, s, lo, lu, th, fees, rebates)
            execd = np.minimum(split, s)
            cash += np.dot(execd, p+fees) - np.dot(split-execd, rebates)
            rem -= execd.sum()
        filled = order_size - max(rem,0)
        avg = cash/filled
        if cash < best['cost']:
            best = {'lambda_over': lo, 'lambda_under': lu,
                    'theta_queue': th, 'cost': cash, 'avg_price': avg}
    return best

# 6. Main Execution & Reporting
if __name__ == '__main__':
    df_pivot, price_cols, size_cols = load_and_prepare('l1_day.csv')
    order_size = 5000
    fees    = np.zeros(len(price_cols))
    rebates = np.zeros(len(price_cols))

    # Tune router parameters
    best = grid_search(df_pivot, price_cols, size_cols, order_size)
    # Compute baselines with best settings
    cash_ba, avg_ba = run_best_ask(df_pivot, price_cols, size_cols, order_size, fees, rebates)
    cash_tw, avg_tw = run_twap(df_pivot, price_cols, size_cols, order_size, fees, rebates)
    cash_vw, avg_vw = run_vwap(df_pivot, price_cols, size_cols, order_size, fees, rebates)

    # Prepare report
    savings = {
        'vs_best_ask': (avg_ba  - best['avg_price'])*1e4/best['avg_price'],
        'vs_twap':     (avg_tw  - best['avg_price'])*1e4/best['avg_price'],
        'vs_vwap':     (avg_vw  - best['avg_price'])*1e4/best['avg_price']
    }
    report = {
        'best_params': {
            'lambda_over':   best['lambda_over'],
            'lambda_under':  best['lambda_under'],
            'theta_queue':   best['theta_queue']
        },
        'router':   {'cash_spent': best['cost'], 'avg_price': best['avg_price']},
        'best_ask': {'cash_spent': cash_ba,  'avg_price': avg_ba},
        'twap':     {'cash_spent': cash_tw,  'avg_price': avg_tw},
        'vwap':     {'cash_spent': cash_vw,  'avg_price': avg_vw},
        'savings_bps': savings
    }
    print(json.dumps(report, indent=2))

    # Plot cumulative cost vs best-ask
    times = []
    cost_r, cost_b = [], []
    rem_r, rem_b = order_size, order_size
    cr, cb = 0.0, 0.0
    for ts, row in df_pivot.iterrows():
        if rem_r<=0 and rem_b<=0: break
        times.append(ts)
        p = row[price_cols].values; s = row[size_cols].values
        split = allocate(rem_r, p, s,
                         best['lambda_over'], best['lambda_under'], best['theta_queue'],
                         fees, rebates)
        exec_r = np.minimum(split, s)
        cr += np.dot(exec_r, p+fees)
        rem_r -= exec_r.sum()
        cost_r.append(cr)
        i = np.argmin(p)
        t = min(rem_b, s[i])
        cb += t * p[i]
        rem_b -= t
        cost_b.append(cb)
    plt.plot(times, cost_r, label='Router', linestyle='--')
    plt.plot(times, cost_b, label='Best-ask', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Cash')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_cost.png')

