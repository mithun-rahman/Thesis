package in.olc;


import java.io.IOException;
import java.io.InvalidObjectException;
import java.util.Iterator;
import java.util.StringTokenizer;

import java.util.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*; 
import org.apache.hadoop.mapred.*; 
import org.apache.hadoop.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;


import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TestMiniMRClientCluster.MyReducer;

public class thesis {
	public static int x = 0;
	public static class MyMap extends MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable>
	{
		@Override
		public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {
			
			String[] rows = value.toString().split(",");
			if(x==1) {
				String prd, cont, mn, yr, dd;
				prd = rows[1];
				cont = rows[7];
				String date[] = rows[4].split(" ");
				String tt[] = date[0].split("/");
				mn = tt[0];
				yr = "20"+tt[2];
				dd = prd+"_"+cont+"_"+mn+"_"+yr;
				int q, p;
				float pr;
				q = Integer.parseInt(rows[3]);
				pr = Float.parseFloat(rows[5]);
				p = (int)pr + 1;
				output.collect(new Text(dd), new IntWritable(p*q));
			}
			else {
				x = 1;
			}
		}
		
	}
	
	public static class MyReducer extends MapReduceBase implements Reducer<Text,IntWritable, Text, IntWritable>{
		
		

		public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text, IntWritable> output,
				Reporter reporter) throws IOException {
			
			int sum = 0;
			while(values.hasNext()) {
				sum += values.next().get();
			}
			output.collect(key, new IntWritable(sum));
		}
	}
	
	public static void main(String[] args) throws Exception {
		JobConf conf = new JobConf(WordCount.class);
		conf.setJobName("mithu");
		
		conf.setMapperClass(MyMap.class);
		conf.setReducerClass(MyReducer.class);
		
		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);
		
		conf.setMapOutputKeyClass(Text.class);
		conf.setMapOutputValueClass(IntWritable.class);
		
		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));
		
		JobClient.runJob(conf);
	}
}
