package hadoopml.libfileinput;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;


public class NLineFileInputFormat extends FileInputFormat<LongWritable, Text> {
    
    @Override
    public RecordReader<LongWritable, Text> getRecordReader(InputSplit split,
            JobConf conf, Reporter reporter) throws IOException {
        return new NLineFileRecordReader(conf, (FileSplit) split);
    }
}

