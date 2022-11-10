lr=0.005
wd=0.0005

for dat in Computers Photo
do
	for hid in 8 16 32 64 128
	do
		python ori_gat.py --dataset=$dat --lr=$lr --weight_decay=$wd --hidden $hid
	done
done

