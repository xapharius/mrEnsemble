package hadoopml.libfileinput.tests;

import hadoopml.libfileinput.NWholeFileRecordReader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.CombineFileSplit;
import org.junit.Assert;
import org.junit.Test;

public class NWholeFileRecordReaderTest {


	@Test
	public void testReadSingleImage() throws IOException {
		File file = new File("../data/test-images/cat_programming.jpg");
		Path path = new Path(file.getPath());
		Path[] paths = new Path[] { path };
		JobConf conf = new JobConf();
		conf.setInt(NWholeFileRecordReader.NUMBER_OF_FILES_PARAM_NAME, 1);
		long[] starts = new long[] { 0 };
		CombineFileSplit split = new CombineFileSplit(conf, paths, starts);
		NWholeFileRecordReader reader = new NWholeFileRecordReader(split, conf);
		Text key = reader.createKey();
		BytesWritable value = reader.createValue();
		reader.next(key, value);
		
		FileInputStream in = new FileInputStream(file);
		int available = in.available();
		byte[] expectedBytes = new byte[available];
		IOUtils.readFully(in, expectedBytes, 0, available);
		int bytesLength = expectedBytes.length;
		expectedBytes = ByteBuffer
				.allocate(4 + bytesLength)
				.putInt(bytesLength)
				.put(expectedBytes)
				.array();
		
		Assert.assertArrayEquals(expectedBytes, value.getBytes());
	}
	
	@Test
	public void testReadMultipleImages() throws IOException {
		List<Path> paths = new ArrayList<Path>(2);
		File file = new File("../data/test-images/cat_programming.jpg");
		paths.add(new Path(file.getPath()));
		file = new File("../data/test-images/cat_programming.jpg");
		paths.add(new Path(file.getPath()));
		JobConf conf = new JobConf();
		conf.setInt(NWholeFileRecordReader.NUMBER_OF_FILES_PARAM_NAME, 2);
		long[] starts = new long[] { 0, 0 };
		CombineFileSplit split = new CombineFileSplit(conf, paths.toArray(new Path[2]), starts);
		NWholeFileRecordReader reader = new NWholeFileRecordReader(split, conf);
		Text key = reader.createKey();
		BytesWritable value = reader.createValue();
		reader.next(key, value);
		
		FileInputStream in = new FileInputStream(file);
		int available = in.available();
		byte[] expectedBytes = new byte[available];
		IOUtils.readFully(in, expectedBytes, 0, available);
		int bytesLength = expectedBytes.length;
		expectedBytes = ByteBuffer
				.allocate((4 + bytesLength)*2)
				.putInt(bytesLength)
				.put(expectedBytes)
				.putInt(bytesLength)
				.put(expectedBytes)
				.array();
		
		Assert.assertArrayEquals(expectedBytes, value.getBytes());
	}
}
