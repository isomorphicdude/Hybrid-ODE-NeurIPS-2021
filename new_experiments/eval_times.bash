cd "$(dirname "$0")/.."  # cd to repo root.

echo "Evaluating times"
python -u -m new_experiments.eval_times >> new_experiments/400_eval_times.txt
echo "Done evaluating times"