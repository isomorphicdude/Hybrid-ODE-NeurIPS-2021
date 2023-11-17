cd "$(dirname "$0")/.."  # cd to repo root.

mkdir -p data

echo "Creating data for training"
python -u -m generated_data.generate_data_train

echo "Creating data for testing"
python -u -m generated_data.generate_data_test

echo "Creating data for noise level=0.4"
python -u -m generated_data.generate_data_noise --noise_level=0.4

echo "Creating data for noise level=0.8"
python -u -m generated_data.generate_data_noise --noise_level=0.8

echo "Creating data for noise level=1.0"
python -u -m generated_data.generate_data_noise --noise_level=1.0

echo "Creating data dim8"
python -u -m generated_data.generate_data_dim8

echo "Creating data dim12"
python -u -m generated_data.generate_data_dim12

# print message when finished
echo "Finished creating data"
