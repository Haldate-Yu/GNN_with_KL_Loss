lr=0.005
wd=0.0005
output_heads=8

for dat in Actor chameleon Cornell Texas Wisconsin squirrel; do
  for hid in 16 32; do
    for alpha1 in 0.01 0.1 1.0; do
      for alpha2 in 0.01 0.1 1.0; do
        for topk in 1 10 20; do
          for hid_mlp in 16 32; do
            python new_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden $hid --output_heads $output_heads --kl_loss True --kl_alpha1 $alpha1 --kl_alpha2 $alpha2 --attn_topk $topk --hidden_mlp $hid_mlp
          done
        done
      done
    done
  done
done
