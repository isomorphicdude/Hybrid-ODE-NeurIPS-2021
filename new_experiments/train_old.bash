# Train old model with sample size = 100, 400, 800

cd "$(dirname "$0")/.."  # cd to repo root.

echo "Train old model with sample size = 100, 400, 800"
python -u -m new_experiments.train_old --sample_size 100 
echo "--------------------------"
python -u -m new_experiments.train_old --sample_size 400
echo "--------------------------"
python -u -m new_experiments.train_old --sample_size 800
echo "Done"