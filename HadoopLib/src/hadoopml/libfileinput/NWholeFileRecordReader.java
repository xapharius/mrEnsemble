package hadoopml.libfileinput;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.codec.binary.Base64;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.lib.CombineFileSplit;

public class NWholeFileRecordReader implements
		RecordReader<Text, BytesWritable> {

	public static final String NUMBER_OF_FILES_PARAM_NAME = "hadoopml.fileinput.filespermap";

	private int numberOfFiles;
	private CombineFileSplit split;
	private int position = 0;
	private JobConf conf;

	public NWholeFileRecordReader(CombineFileSplit split, JobConf conf)
			throws IOException {
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
	public BytesWritable createValue() {
		return new BytesWritable();
	}

	@Override
	public Text createKey() {
		return new Text();
	}

	@Override
	public long getPos() throws IOException {
		return (long) (getProgress()*this.split.getLength());
	}

	@Override
	public float getProgress() throws IOException {
		return this.position / numberOfFiles;
	}

	@Override
	public boolean next(Text key, BytesWritable value) throws IOException {
		Path[] paths = split.getPaths();
		// test if there are more files available
		if (this.position >= paths.length) {
			key.set("");
			value.setSize(0);
			return false;
		}
		StringBuilder keyBuilder = new StringBuilder();
		List<byte[]> fileData = new LinkedList<byte[]>();
		int fullDataLength = 0;
		// try to read numberOfFiles images
		for (int i = this.position; i < numberOfFiles; i++) {
			// if there are no images left we have to stop
			if (i >= split.getNumPaths()) {
				this.position = this.numberOfFiles;
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
				in = fs.open(file);
				int available = in.available();
				byte[] data = new byte[available];
				// read full file
				IOUtils.readFully(in, data, 0, available);
				int dataLength = data.length;
				// encode each file as [ 4 bytes length | file bytes ]
				// prepend length of file
				data = ByteBuffer
						.allocate(4 + dataLength)
						.putInt(dataLength)
						.put(data)
						.array();
				fileData.add(data);
				fullDataLength += data.length;
			} finally {
				if (in != null)
					in.close();
			}
			this.position++;
		}
		// put all data together
		ByteBuffer buffer = ByteBuffer.allocate(fullDataLength);
		for (byte[] bytes : fileData) {
			buffer.put(bytes);
		}
		byte[] encoded = Base64.encodeBase64(buffer.array());
		value.setCapacity(encoded.length);
		value.set(encoded, 0, encoded.length);
		key.set(keyBuilder.toString());
		
		return true;
	}
}
