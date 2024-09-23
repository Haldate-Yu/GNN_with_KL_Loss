lr=0.005
wd=0.0005

for dat in Actor chameleon Cornell Texas Wisconsin squirrel
do
	for hid in 16 32
	do
		python ori_gatv2.py --data=$dat --lr=$lr --weight_decay=$wd --hidden $hid
	done
done

