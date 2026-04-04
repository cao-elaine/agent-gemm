#!/bin/bash
# Auto-restart wrapper for the SmolVLA NPU experiment.
# Runs the Claude agent repeatedly until all 291 shapes are complete.
# Each restart picks up where the last left off (resume logic in the prompt).

PROMPT="Read and execute the prompt at prompts/agent-smolvla-experiment.md with these inputs:

dtype       = bf16
trials      = 3
shapes_file = references/smol-vla-dataset/smolvla_gemm_shapes.json"

TOTAL_SHAPES=291
DATE_STR=$(date +%Y%m%d)
RESULTS_FILE="references/experiment-results/test_n_${DATE_STR}/results.json"

cd /home/ec935/agent-gemm

run=1
while true; do
    echo ""
    echo "============================================"
    echo " SmolVLA Experiment — Run #${run}"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    # Check how many shapes are done
    if [ -f "$RESULTS_FILE" ]; then
        done=$(python3 -c "
import json
data = json.load(open('$RESULTS_FILE'))
print(sum(1 for s in data.get('shapes',[]) if s.get('winner') is not None))
" 2>/dev/null || echo "0")
        echo " Completed so far: ${done} / ${TOTAL_SHAPES}"
        if [ "$done" -ge "$TOTAL_SHAPES" ]; then
            echo " All shapes complete! Exiting."
            break
        fi
    fi

    # Run the agent (non-interactive)
    claude --dangerously-skip-permissions -p "$PROMPT"

    echo ""
    echo " Agent exited (likely context limit). Restarting in 5 seconds..."
    sleep 5
    run=$((run + 1))
done

echo ""
echo "Experiment complete. Results at: $RESULTS_FILE"
