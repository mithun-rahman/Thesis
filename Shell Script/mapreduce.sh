
#!usr/bin/bash

function Make_Directory()
{
	echo "Input Directory Name: " 
	read dir
	hadoop fs -mkdir /$dir
}

function Put_Data()
{
	echo "Dataset Name: " 
	read data
	hadoop fs -put ~/hadoopMyFiles/$data /$dir
}

function Start_MapReduce()
{
	echo "Output Directory Name: "  
	read out
	#hadoop jar ~/hadoopMyFiles/prd_cont_mnyr.jar /$dir/$data /OutPut25
	hadoop jar ~/hadoopMyFiles/prd_cont_mnyr.jar /$dir/$data /$out
}


function Print_Output()
{
	echo "               ________OUTPUT_________"
	#hadoop fs -cat /$out/part-00000
	cat part-00000
}

Make_Directory
Put_Data
#Start_MapReduce
Print_Output
