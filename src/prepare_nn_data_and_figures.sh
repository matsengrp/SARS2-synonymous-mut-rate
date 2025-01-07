
echo "Calculating R^2 for the generalized linear model on train-test splits..."

python get_r2_for_basel_models.py

echo "R^2 values written to results/basel_r2s.csv"
echo "Train-test splits written to results/split_syn_mut_counts.csv"


echo "Training NN models on train-test splits and calculating R^2 (this may take awhile)..."
python make_r2_plots.py

echo ""
echo "R^2 values written to results/combined_results.csv"
echo "Box plots of R^2 values written to results/r2_comparison.pdf"