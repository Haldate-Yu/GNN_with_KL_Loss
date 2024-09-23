lr=0.005
wd=0.0005
mx=MX
sd=SD

for dat in Actor chameleon Cornell Texas Wisconsin squirrel; do
  python ori_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --runs 1 --epochs 1
  python ori_gatv2.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --runs 1 --epochs 1
  python ori_super_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --attention_type $mx --runs 1 --epochs 1
  python ori_super_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --attention_type $sd --runs 1 --epochs 1

  python new_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --output_heads 8 --kl_loss True \
    --kl_alpha1 0.1 --kl_alpha2 0.1 --attn_topk 10 --hidden_mlp 8 --runs 1 --epochs 1

  python new_gatv2.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --output_heads 8 --kl_loss True \
    --kl_alpha1 0.1 --kl_alpha2 0.1 --attn_topk 10 --hidden_mlp 8 --runs 1 --epochs 1

  python new_super_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --output_heads 8 --kl_loss True \
    --kl_alpha1 0.1 --kl_alpha2 0.1 --attn_topk 10 --hidden_mlp 8 --attention_type $mx --runs 1 --epochs 1

  python new_super_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden 8 --output_heads 8 --kl_loss True \
    --kl_alpha1 0.1 --kl_alpha2 0.1 --attn_topk 10 --hidden_mlp 8 --attention_type $sd --runs 1 --epochs 1
done
