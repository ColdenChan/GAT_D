#!/bin/bash
 
#####################################################
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
#####################################################
# File:    sh_execute_dense.sh
# Author:  cb
# Date:    2018/12/20 14:21:42
# Brief:
#####################################################

i=1
j=1
k=1
end_i=$1	#2-nd mask element
end_j=$2	#run times
end_k=$3	#program nums every run
while(( $i<=$end_1 ))
do
	while (( $j<=$end_2 ))
	do
		j=1
		while(( $k<=$end_3 ))
		do
			time=`date "+%Y%m%d%H%M%S"`
			python execute_dense.py --runid $k --t $i-1 --dataset ${4} --gpuid ${5} > ./log/cora/${time}.log 2>&1 &
			sleep 5 
			let k++
		done
		sleep 600
		let j++
	done
	sleep 30
	let i++
done


# vim: set expandtab ts=4 sw=4 sts=4 tw=100
