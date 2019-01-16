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
end=$1
end2=$2
while(( $i<=$end ))
do
	j=1
	while(( $j<=$end2 ))
	do
		time=`date "+%Y%m%d%H%M%S"`
		python execute_dense.py --runid $j --dataset ${3} --gpuid ${4} > ./log/cora/${time}.log 2>&1 &
		sleep 5 
		let j++
	done
	sleep 600
	let i++
done


# vim: set expandtab ts=4 sw=4 sts=4 tw=100
