package hadoopml.libfileinput.tests;

import hadoopml.libfileinput.util.ByteUtil;

import org.junit.Assert;
import org.junit.Test;

public class ByteUtilTest {

	@Test
	public void testReplace() {
		byte[] arr = { 0x11, 0x22, 0x0a, 0x33, 0x0a };
		byte toFind = 0x0a;
		byte[] replaceWith = { 0x00, 0x00 };
		byte[] exp = { 0x11, 0x22, 0x00, 0x00, 0x33, 0x00, 0x00 };
		byte[] is = ByteUtil.replace(arr, toFind, replaceWith);
		
		Assert.assertArrayEquals(exp, is);
	}

}
