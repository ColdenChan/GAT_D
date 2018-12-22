#!/bin/bash
 
#####################################################
# Copyright (c) 2018 USTC, Inc. All Rights Reserved
#####################################################
# File:    sh_execute_sparse.sh
# Author:  cb
# Date:    2018/12/20 14:21:42
# Brief:
#####################################################

i=1
end=$1
echo $end
while(( i<=$end ))
do
	time=`date "+%Y%m%d%H%M%S"`
	python execute_sparse.py --dataset ${2} --gpuid ${3} > ./log/cora/${time}.log 2>&1 &
	sleep 1
	let i++
done

# vim: set expandtab ts=4 sw=4 sts=4 tw=100
