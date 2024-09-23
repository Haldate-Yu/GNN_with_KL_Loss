lr=0.005
wd=0.0005
attn_type=SD

for dat in Actor chameleon Cornell Texas Wisconsin squirrel
do
	for hid in 16 32
	do
		python ori_super_gat.py --data=$dat --lr=$lr --weight_decay=$wd --hidden $hid --attention_type $attn_type
	done
done

