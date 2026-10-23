#!/bin/bash
# Monitor the automated pipeline progress

echo "==================================================================="
echo "MONITORING AUTOMATED QWEN UNFREEZING PIPELINE"
echo "==================================================================="
echo ""

# Find latest log
LATEST_LOG=$(ls -t data/auto_pipeline_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "No pipeline log found."
    exit 1
fi

echo "Log file: $LATEST_LOG"
echo ""

# Check if process is running
if pgrep -f "auto_unfreeze_qwen_pipeline.py" > /dev/null; then
    echo "Status: 🟢 RUNNING"
else
    echo "Status: ⚪ FINISHED or NOT RUNNING"
fi
echo ""

# Show latest training progress
echo "-------------------------------------------------------------------"
echo "TRAINING PROGRESS (last 10 lines):"
echo "-------------------------------------------------------------------"
tail -10 "$LATEST_LOG" | grep -E "\[.*\]|val_loss|BEST|PHASE|RESULT|COMPLETE" || tail -10 "$LATEST_LOG"
echo ""

# Find output directory
OUTPUT_DIR=$(grep -o "data/ur_qwen_unfrozen_auto_[0-9_]*" "$LATEST_LOG" | head -1)

if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    echo "-------------------------------------------------------------------"
    echo "FILES IN OUTPUT: $OUTPUT_DIR"
    echo "-------------------------------------------------------------------"
    ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -10
    echo ""

    # Check if evaluation started
    if [ -d "$OUTPUT_DIR/evaluation" ]; then
        echo "-------------------------------------------------------------------"
        echo "EVALUATION RESULTS:"
        echo "-------------------------------------------------------------------"
        if [ -f "$OUTPUT_DIR/evaluation/result.json" ]; then
            cat "$OUTPUT_DIR/evaluation/result.json" | python3 -m json.tool | head -20
        else
            echo "Evaluation in progress... ($(ls $OUTPUT_DIR/evaluation/episode_*.json 2>/dev/null | wc -l) episodes completed)"
        fi
        echo ""
    fi

    # Show final summary if exists
    if [ -f "$OUTPUT_DIR/pipeline_summary.json" ]; then
        echo "-------------------------------------------------------------------"
        echo "PIPELINE SUMMARY:"
        echo "-------------------------------------------------------------------"
        cat "$OUTPUT_DIR/pipeline_summary.json" | python3 -m json.tool
        echo ""
    fi
fi

echo "==================================================================="
echo "To view full log: tail -f $LATEST_LOG"
echo "To stop pipeline: pkill -9 -f auto_unfreeze_qwen_pipeline.py"
echo "==================================================================="
