package hadoopml.libfileinput;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.LineRecordReader;
import org.apache.hadoop.mapred.RecordReader;


public class NLineFileRecordReader implements RecordReader<LongWritable, Text> {

    public static final String NUMBER_OF_LINES_PARAM_NAME = "hadoopml.fileinput.linespermap";
    
    private LineRecordReader lineReader;
    private LongWritable currentKey;
    private Text currentValue;
    private int numberOfLines;
    
    public NLineFileRecordReader(JobConf conf, FileSplit split) throws IOException {
        numberOfLines = conf.getInt(NUMBER_OF_LINES_PARAM_NAME, 1);
        lineReader = new LineRecordReader(conf, split);
        currentKey = lineReader.createKey();
        currentValue = lineReader.createValue();
        System.err.println("Number of lines: " + numberOfLines);
    }
    
    @Override
    public void close() throws IOException {
        lineReader.close();
    }

    @Override
    public Text createValue() {
        return lineReader.createValue();
    }

    @Override
    public LongWritable createKey() {
        return lineReader.createKey();
    }

    @Override
    public long getPos() throws IOException {
        return lineReader.getPos();
    }

    @Override
    public float getProgress() throws IOException {
        return lineReader.getProgress();
    }
    
    @Override
    public boolean next(LongWritable key, Text value) throws IOException {
        if (!lineReader.next(currentKey, currentValue)) {
            key.set(0);
            value.set("");
            return false;
        }
        LongWritable localKey = currentKey;
        Text localValue = lineReader.createValue();
        // append lines until specified number is reached
        for (int i = 0; i < numberOfLines-1; i++) {
            // abort when there are no more lines
            if (!lineReader.next(localKey, localValue)) {
                key.set(localKey.get());
                value.set(currentValue.getBytes());
                return true;
            }
            // append line
            byte[] bytes = new byte[localValue.getLength() + 2];
            bytes[0] = '\\';
            bytes[1] = 'n';
            System.arraycopy(localValue.getBytes(), 0, bytes, 2, localValue.getLength());
            byte[] result = new byte[currentValue.getLength() + bytes.length];
            System.arraycopy(currentValue.getBytes(), 0, result, 0, currentValue.getLength());
            System.arraycopy(bytes, 0, result, currentValue.getLength(), bytes.length);
            currentValue.set(result);
        }
        key.set(localKey.get());
        value.set(currentValue.getBytes());
        return true;
    }
}
