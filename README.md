**Cont–Kukanov Multi-Venue Order Routing Backtest**

**Purpose**

Simulate execution of a single 5000-share buy order across multiple trading venues, comparing a smart Cont–Kukanov allocator against three benchmarks (Best-Ask, TWAP, VWAP). The goal is to minimize total cost—cash outlay, maker rebates, under/over-fill penalties and queue-risk—by optimally splitting volume.

**Code Structure**

1\. Data Prep: Load Level-1 quotes, clean timestamps, dedupe per venue, pivot into per-venue price/size columns, fill missing values.

2\. Cost Function: Compute cash spent (price+fee), rebates on unfilled shares, penalties for under/over-execution and queue risk.

3\. Allocator: For each market update, brute-force candidate splits (100-share steps), score via cost function, choose the lowest-cost allocation, simulate fills, update remaining shares.

4\. Baselines:

Best-Ask: always trade at the cheapest live quote. 

TWAP: slice evenly over 60 s buckets (≥1 share/bucket), split by displayed size.

VWAP: split per snapshot according to displayed size weights.

5\. Grid Search: Tune λ\_over, λ\_under, θ\_queue over {0.01, 0.05, 0.10, 0.20}³ to find minimal total cost.

6\. Reporting & Plot: Output JSON with best parameters, costs/prices for router and baselines, basis-point savings; save a cumulative-cost chart (router vs. Best-Ask).

**Search Choices**

We tune the three penalty parameters by a grid search over {0.01, 0.05, 0.10, 0.20} for each of λ\_over, λ\_under, and θ\_queue (64 combinations). These values reflect light to moderate aversion to under-/over-fills and queue risk, and the coarse grid executes in seconds. Once the best region is found, one could refine to finer increments around the optimum.

**Results**

In a single-venue test all strategies coincide (zero bps savings); the plot shows overlapping router and Best-Ask curves.

**Future Improvements**

Deeper Market Data: Extend the model to use deeper order book levels and simulate partial fills over multiple price levels. This would allow the allocator to consider liquidity beyond the top of the book and truly optimize large orders that may sweep through several price tiers.

Adaptive Parameters & Execution Risk: Introduce a dynamic or more sophisticated model for fill probability (queue risk) and adapt theta\_queue over time. For example, one could incorporate a probabilistic execution model or learning-based parameter tuning so that the strategy better adapts to changing market conditions (reducing the risk of unfilled orders while avoiding unnecessary aggressive fills).

