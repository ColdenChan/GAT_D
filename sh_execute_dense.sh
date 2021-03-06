#!/bin/bash
 
#####################################################
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
#####################################################
# File:    sh_execute_dense.sh
# Author:  cb
# Date:    2018/12/20 14:21:42
# Brief:
#####################################################

i=1000
j=1
k=1
end_i=$1	#2-nd mask element
end_j=$2	#run times
end_k=$3	#program nums every run
while(( $i<=$end_i ))
do
	if [ ! -d "./log/${4}/${i}" ]; then
		mkdir ./log/${4}/${i}
	fi
	echo -$i
	j=1
	while(( $j<=$end_j ))
	do
		k=1
		while(( $k<=$end_k ))
		do
			time=`date "+%Y%m%d%H%M%S"`
			# if [ -d "./log/${4}/${i}" ]; then
				# rm -rf ./log/${4}/${i}
			# fi
			python execute_dense.py --runid $k --t -$i --dataset ${4} --gpuid ${5} > ./log/${4}/${i}/${time}.log 2>&1 &
			sleep 3
			let k++
		done
		sleep 240
		let j++
	done
	sleep 10
	((i+=1000))
done


# vim: set expandtab ts=4 sw=4 sts=4 tw=100
