package hadoopml.libfileinput;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.CombineFileInputFormat;
import org.apache.hadoop.mapred.lib.CombineFileSplit;
 
public class NWholeFileInputFormat extends CombineFileInputFormat<Text, BytesWritable> {
    
    @Override
    protected boolean isSplitable(FileSystem fs, Path filename) {
        return false;
    }

    @Override
    public RecordReader<Text, BytesWritable> getRecordReader(
      InputSplit split, JobConf job, Reporter reporter) throws IOException {
        return new NWholeFileRecordReader((CombineFileSplit) split, job);
    }
}
