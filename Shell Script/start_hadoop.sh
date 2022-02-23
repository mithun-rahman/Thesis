
#!usr/bin/bash

function Hadoop_INIT()
{
	~/Downloads/hadoop-3.2.1/bin/hdfs namenode -format
	export PDSH_RCMD_TYPE=ssh
	~/Downloads/hadoop-3.2.1/sbin/start-dfs.sh
	~/Downloads/hadoop-3.2.1/sbin/start-yarn.sh
}

Hadoop_INIT

echo "/************************************************************/"
echo "Welcome to Hadoop"
echo "170213 and 170225"
echo "CSE, Khulna University"
echo "Local Host:9870 (HDFS)"
echo "Local Host:8088 (YARN)"
echo "************************************************************/"
