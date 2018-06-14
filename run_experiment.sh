declare -a corpora=("state_union")
declare -a regularizations=("prev_abs", "L1", "L2", "none")

for corpus in "${corpora[@]}"
do
  for reg in "${regularizations[@]}"
  do
    python train_embeddings.py --corpus=$corpus \
      --regularization=$reg --reg_weight=0.01 \
      --num_epochs=200 --max_dim=8 \
      --num_runs=5
  done
done
