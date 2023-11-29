# run and direct results to new_experiments/constant_coeff.txt
cd "$(dirname "$0")/.."  # cd to repo root.
echo "Running constant coefficient experiments"
python -u -m new_experiments.run_constant_coeff >> new_experiments/constant_coeff.txt
echo "Done running constant coefficient experiments"