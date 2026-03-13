# Context: Padding Heuristics and Profiling Data

## Overview
This directory contains a set of JSON files detailing the profiling, comparison, and resulting heuristics for padding unaligned matrix dimensions before execution on an NPU. The files evaluate two distinct software padding implementations: `manual_copy` and `numpy_pad`. The sweep files have been cleaned and grouped by `input_shape` to minimize repetition and optimize parsing.

---

## 1. `padding_sweep_copy.json`

### Description
This file contains the cleaned profiling data for the **`manual_copy`** padding implementation. It groups various padding target shapes under their original input shapes to show how long it takes to manually copy elements into larger memory boundaries.

### Field Breakdown
* **`input_shape`** (array): The raw, unpadded [M, N, K] dimensions of the starting matrices.
* **`padding_results`** (array): A list of evaluated padding configurations for this specific input shape.
  * **`dtype`** (string): The data type used for the operation (e.g., "i8", "i16", "bf16").
  * **`padding_shape`** (array): The target [M, N, K] dimensions the input was padded to.
  * **`padding_us`** (float): The time taken to execute the padding via `manual_copy` (in microseconds).
  * **`tile`** (array): The execution tile size [m, n, k] selected for the NPU (based on profiling data in ../npu_execution_profiling.json).

### Significance for Agent Decision Making
Provides the agent with direct lookup tables for memory-bound latency. The agent can use this to understand how the `manual_copy` fallback scales as the difference between the `input_shape` and `padding_shape` grows.

---

## 2. `padding_sweep_numpy.json`

### Description
This file mirrors the structure of the manual copy sweep but contains the profiling results for the **`numpy_pad`** implementation (using NumPy's underlying C-backend).

### Field Breakdown
* **`input_shape`** (array): The raw, unpadded [M, N, K] dimensions.
* **`padding_results`** (array): A list of evaluated padding configurations.
  * **`dtype`** (string): The data type (e.g., "i8", "i16", "bf16").
  * **`padding_shape`** (array): The target [M, N, K] dimensions the input was padded to.
  * **`padding_us`** (float): The time taken to execute the padding via `numpy_pad` (in microseconds).
  * **`tile`** (array): The execution tile size [m, n, k] selected for the NPU (based on profiling data in ../npu_execution_profiling.json).

### Significance for Agent Decision Making
Allows the agent to evaluate NumPy's C-bindings performance. By comparing `padding_us` between this file and the `manual_copy` file, the agent can observe that NumPy is highly inefficient for small tensor shapes but overtakes manual copying on massive shapes due to vectorized optimizations.

---

## 3. `padding_transition_db.json`

### Description
This file acts as a synthesized "Head-to-Head" database. It isolates the absolute best padding times from both sweeps for a given input shape and declares which software implementation won.

### Field Breakdown
* **`num_unique_shapes`** (int): The total number of grouped input shapes evaluated.
* **`total_records_processed`** (int): The total number of distinct matrix/dtype combinations evaluated.
* **`results`** (array): The list of aggregated comparison objects.
  * **`input_shape`** (array): The original [M, N, K] dimensions.
  * **`comparisons`** (array): The head-to-head results grouped by data type.
    * **`dtype`** (string): The data type evaluated.
    * **`manual_copy_time_us`** (float): The lowest padding time achieved using `manual_copy`.
    * **`numpy_pad_time_us`** (float): The lowest padding time achieved using `numpy_pad`.
    * **`fastest_padding_method`** (string): The definitive winner ("manual_copy" or "numpy_pad").
    * **`manual_best_padded_shape`** (array): The target shape that yielded the fastest manual time.
    * **`numpy_best_padded_shape`** (array): The target shape that yielded the fastest NumPy time.

### Significance for Agent Decision Making
This is the ground-truth training data. If an agent needs to build, debug, or verify a heuristic routing algorithm, this file provides the absolute correct answer for which padding implementation should be chosen for any known matrix shape.

---

## 4. `padding_transition_thresholds.json`

### Description
This file contains the final, compiled $O(1)$ decision boundaries derived from the transition DB. It provides the exact mathematical cutoffs for when to switch padding strategies.

### Field Breakdown
* **`feature`** (string): The mathematical metric used to measure tensor size (e.g., `ab_elements`, calculated as $M \times K + K \times N$).
* **`dtype_thresholds`** (object): The crossover boundaries categorized by data type.
  * **`num_samples`** (int): The dataset size used to generate this threshold.
  * **`small_strategy`** (string): The required padding function when the feature is *less than* the threshold (e.g., "manual_copy").
  * **`large_strategy`** (string): The required padding function when the feature is *greater than* the threshold (e.g., "numpy_pad").
  * **`threshold`** (int): The critical feature value where the optimal strategy flips.
  * **`avg_chosen_us`** (float): The expected average padding time when executing this heuristic across the dataset.
  * **`avg_regret_us`** (float): The average time penalty incurred by using this static threshold instead of a perfect oracle.

### Significance for Agent Decision Making
**This is the active runtime policy.** When the agent is writing or refactoring compiler/runtime logic, it should use this file to implement its routing conditions. (e.g., *if `dtype` == "i8" and `ab_elements` < 1,653,750, route to `manual_copy`*).

## Entries added by agent runs

(none yet — if the agent believes that the cutoff threshold between `manual_copy` and `numpy_pad` should be updated, the agent can update padding_transition_thresholds.json)