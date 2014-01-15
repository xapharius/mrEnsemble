package hadoopml.libfileinput.newapi;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;


public class NLineFileRecordReader extends RecordReader<LongWritable, Text> {

	private static final String NUMBER_OF_LINES_PARAM_NAME = "hadoopml.fileinput.linespermap";
	
	private LineRecordReader lineReader;
	private LongWritable currentKey;
	private Text currentValue;
	private int numberOfLines;
	
	@Override
	public void close() throws IOException {
		lineReader.close();
	}

	@Override
	public LongWritable getCurrentKey() throws IOException, InterruptedException {
		return currentKey;
	}

	@Override
	public Text getCurrentValue() throws IOException, InterruptedException {
		return currentValue;
	}

	@Override
	public float getProgress() throws IOException, InterruptedException {
		return lineReader.getProgress();
	}

	@Override
	public void initialize(InputSplit split, TaskAttemptContext context)
			throws IOException, InterruptedException {
		numberOfLines = context.getConfiguration().getInt(NUMBER_OF_LINES_PARAM_NAME, 1);
		lineReader = new LineRecordReader();
		lineReader.initialize(split, context);
	}

	@Override
	public boolean nextKeyValue() throws IOException, InterruptedException {
		if (!lineReader.nextKeyValue()) {
			return false;
		}
		currentKey = lineReader.getCurrentKey();
		Text line = lineReader.getCurrentValue();
		currentValue = line;
		// append lines until specified number is reached
		for (int i = 0; i < numberOfLines-1; i++) {
			// abort when there are no more lines
			if (!lineReader.nextKeyValue()) {
				return true;
			}
			byte[] bytes = new byte[line.getLength() + 1];
			bytes[0] = '\n';
			System.arraycopy(line.getBytes(), 0, bytes, 1, line.getBytes().length);
			currentValue.append(bytes, 0, bytes.length);
		}
		return true;
	}
}
