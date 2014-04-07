package hadoopml.libfileinput.tests;

import hadoopml.libfileinput.NWholeFileRecordReader;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;

import org.apache.hadoop.fs.Path;
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
		Text value = reader.createValue();
		reader.next(key, value);
		
		BufferedImage image = ImageIO.read(file);
		int[] expectedPixels = null;
		expectedPixels = image.getData().getPixels(0, 0, image.getWidth(), image.getHeight(), expectedPixels);
		
		String[] lines = value.toString().split("\n");
		int numBands = Integer.parseInt(lines[0]);
		int[] isPixels = new int[image.getWidth() * image.getHeight() * numBands];
		for (int y = 1; y < lines.length; y++) {
			String line = lines[y];
			if (line.equals("\n"))
				continue;
			String[] pixelStrings = line.split(",");
			for (int x = 0; x < pixelStrings.length; x++) {
				String pixelString = pixelStrings[x];
				if (pixelString.equals("\n"))
					continue;
				isPixels[(y-1)*image.getWidth()*numBands + x] = Integer.parseInt(pixelString);
			}
		}
		
		Assert.assertArrayEquals(expectedPixels, isPixels);
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
		long[] starts = new long[] { 0 };
		CombineFileSplit split = new CombineFileSplit(conf, paths.toArray(new Path[2]), starts);
		NWholeFileRecordReader reader = new NWholeFileRecordReader(split, conf);
		Text key = reader.createKey();
		Text value = reader.createValue();
		reader.next(key, value);
		
		BufferedImage image = ImageIO.read(file);
		int[] temp = null;
		temp = image.getData().getPixels(0, 0, image.getWidth(), image.getHeight(), temp);
		Integer[] expectedPixels = new Integer[temp.length];
		
		for (int i = 0; i < temp.length; i++)
			expectedPixels[i] = temp[i];
		
		String[] lines = value.toString().split("\n");
		
		List<Integer[]> images = new ArrayList<Integer[]>(2);
		
		int numBands = 0;
		List<Integer> pixels = null;
		for (int lineNo = 0; lineNo < lines.length; lineNo++) {
			String line = lines[lineNo];
			if (!line.contains(",")) {
				if (pixels != null) {
					Integer[] pixelsArr = new Integer[pixels.size()];
					pixels.toArray(pixelsArr);
					images.add(pixelsArr);
				}
				numBands = Integer.parseInt(lines[lineNo]);
				pixels = new LinkedList<Integer>();
				continue;
			}
			
			String[] pixelStrings = line.split(",");
			for (int x = 0; x < pixelStrings.length; x++) {
				String pixelString = pixelStrings[x];
				if (pixelString.equals("\n"))
					continue;
				pixels.add(Integer.parseInt(pixelString));
			}
		}
		Integer[] pixelsArr = new Integer[pixels.size()];
		pixels.toArray(pixelsArr);
		images.add(pixelsArr);
		
		Assert.assertArrayEquals(expectedPixels, images.get(0));
		Assert.assertArrayEquals(expectedPixels, images.get(1));
	}
}
