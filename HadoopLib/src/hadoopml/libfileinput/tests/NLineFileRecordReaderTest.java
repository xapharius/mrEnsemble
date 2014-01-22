package hadoopml.libfileinput.tests;

import static org.junit.Assert.fail;
import hadoopml.libfileinput.NLineFileRecordReader;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.StringUtils;
import org.junit.Assert;
import org.junit.Test;

public class NLineFileRecordReaderTest {

	
	@Test
	public void testReadSingleLine() throws IOException {
		File file = new File("../data/wine-quality/winequality-red.csv");
		Path path = new Path(file.getPath());
		JobConf conf = new JobConf();
		conf.setInt(NLineFileRecordReader.NUMBER_OF_LINES_PARAM_NAME, 1);
		FileSplit split = new FileSplit(path, 0, file.length(), new String[0]);
		NLineFileRecordReader reader = new NLineFileRecordReader(conf, split);
		LongWritable key = reader.createKey();
		Text value = reader.createValue();
		reader.next(key, value);
		
		String firstLine = new BufferedReader(new FileReader(file)).readLine();
		
		Assert.assertEquals(firstLine, value.toString());
	}

	@Test
	public void testReadMultipleLines() throws IOException {
		Random random = new Random(System.currentTimeMillis());
		int numLoops = random.nextInt(40) + 10;

		for (int i = 0; i < numLoops; i++) {
			int numberOfLines = random.nextInt(180) + 20;
			File file = new File("../data/wine-quality/winequality-red.csv");
			Path path = new Path(file.getPath());
			JobConf conf = new JobConf();
			conf.setInt(NLineFileRecordReader.NUMBER_OF_LINES_PARAM_NAME, numberOfLines);
			FileSplit split = new FileSplit(path, 0, file.length(), new String[0]);
			NLineFileRecordReader reader = new NLineFileRecordReader(conf, split);
			LongWritable key = reader.createKey();
			Text value = reader.createValue();
			reader.next(key, value);
			
			BufferedReader fileReader = new BufferedReader(new FileReader(file));
			String expected = "";
			for (int j = 0; j < numberOfLines; j++) {
				String line = fileReader.readLine();
				expected += line;
				if (j != numberOfLines-1)
					expected += "\\n";
			}
			fileReader.close();
			
			Assert.assertEquals(expected, value.toString());
		}
	}
}
