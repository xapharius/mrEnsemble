package hadoopml.libfileinput;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.lib.CombineFileSplit;
import org.apache.hadoop.util.StringUtils;


public class NWholeFileRecordReader implements RecordReader<Text, Text> {

    public static final String NUMBER_OF_FILES_PARAM_NAME = "hadoopml.fileinput.filespermap";
    
    private int numberOfFiles;
    private CombineFileSplit split;
    private int position = 0;
    private JobConf conf;

    public NWholeFileRecordReader(CombineFileSplit split, JobConf conf) throws IOException {
        this.conf = conf;
        this.split = split;
        this.numberOfFiles = conf.getInt(NUMBER_OF_FILES_PARAM_NAME, 1);
        System.err.println("Number of files: " + numberOfFiles);
    }
    
    @Override
    public void close() throws IOException {
        // do nothing
    }

    @Override
    public Text createValue() {
        return new Text();
    }

    @Override
    public Text createKey() {
        return new Text();
    }

    @Override
    public long getPos() throws IOException {
        return this.position;
    }

    @Override
    public float getProgress() throws IOException {
        return this.position / this.split.getLength();
    }

    @Override
    public boolean next(Text key, Text value) throws IOException {
        Path[] paths = split.getPaths();
        StringBuilder keyBuilder = new StringBuilder();
        StringBuilder valueBuilder = new StringBuilder();
        // try to read numberOfFiles images
        for(int i = position; i < numberOfFiles; i++) {
            // if there are no images left we have to stop
            if (i >= split.getNumPaths()) {
                break;
            }
            Path file = paths[i];
            String fileName = file.getName();
            if (i > 0) {
                keyBuilder.append(",");
            }
            keyBuilder.append(fileName);
            
            FileSystem fs = file.getFileSystem(conf);
            FSDataInputStream in = null;
            try {
                // encode each image as comma separated list of pixel values
                // where a line is a row in the image and images are separated
                // by a line containing just the number of bands of the next 
            	// image.
                in = fs.open(file);
                BufferedImage image = ImageIO.read(in.getWrappedStream());
                int[] arr = null;
                arr = image.getData().getPixels(0, 0, image.getWidth(), image.getHeight(), arr);
                String[] lines = new String[image.getHeight()];
                int numBands = image.getData().getNumBands();
                valueBuilder.append(numBands);
                valueBuilder.append("\n");
                // iterate through rows
                for (int y = 0; y < image.getHeight(); y++) {
                    String[] line = new String[image.getWidth()*numBands];
                    // iterate through columns
                    for (int x = 0; x < image.getWidth()*numBands; x++) {
                        // arr is concatenation of all pixels, hence the pixel
                        // at (x,y) is in arr at y*width + x + b
                        line[x] = Integer.toString(arr[y*image.getWidth()*numBands + x]);
                    }
                    // add the current line, values are comma separated
                    lines[y] = StringUtils.join(",", Arrays.asList(line));
                }
                // add the current image, images are separated by a new line
                valueBuilder.append(StringUtils.join("\n", Arrays.asList(lines)));
                valueBuilder.append("\n");
            } finally {
                in.close();
            }
        }
        value.set(valueBuilder.toString());
        key.set(keyBuilder.toString());
        return true;
    }
}
