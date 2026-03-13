# Context: Baseline 1 Profiling Results (`results.json`)

## Dataset Overview
The `results.json` file in this directory contains the profiling and timing data for **Baseline 1**. 

**Heuristic Definition:** Baseline 1 applies a heuristic where all randomly generated inputs are padded to the **closest supported M,N,K execution size**, rather than evaluating and padding to the M,N,K size that yields the absolute fastest NPU execution time.

---

## Data Schema

### Top-Level Metadata
* `num_cases` (int): The total number of matrix multiplication test cases evaluated.
* `num_results` (int): The total number of completed results logged in this file.
* `num_failed_runtime` (int): The number of cases that encountered a failure during NPU runtime execution.
* `num_failed_selection` (int): The number of cases that failed during the padding strategy selection phase.
* `summary_by_dtype` (list): Aggregate statistics grouped by data type (`i8`, `i16`, `bf16`), detailing the count of cases and the `mean_total_component_us` (average total execution time in microseconds).

### Detailed Results Array (`results`)
Each object in the `results` array represents a single test case with the following fields:

* **Matrix Characteristics:**
  * `dtype` (string): The data type used for the operation (e.g., "i8", "i16", "bf16").
  * `input_shape` (array): The original, raw [M, N, K] tensor dimensions before padding.
  * `padded_shape` (array): The target [M, N, K] dimensions after padding is applied (determined by the closest size heuristic).
  * `tile` (array): The execution tile dimensions [m, n, k] utilized by the NPU.

* **Decision & Processing Timings (in microseconds - `us`):**
  * `select_padding_strategy_us` (float): Time spent determining the overarching padding strategy.
  * `select_padded_shape_us` (float): Time spent calculating the closest valid padded shape.
  * `padding_us` (float): Time spent physically applying the padding to the inputs in memory.
  * `npu_execution_us` (float): Time spent running the actual hardware execution on the NPU.
  * `unpadding_us` (float): Time spent un-padding/extracting the resulting output tensor.
  * `total_time` (float): The total, end-to-end component time for this operation.

* **Implementation Details:**
  * `padding_function` (string): The specific software implementation used to apply the padding (e.g., "manual_copy", "numpy_pad").
  * `passed` (bool): Indicates whether the test case completed successfully without errors.

---

##