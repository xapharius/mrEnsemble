package hadoopml.libfileinput.util;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;

public class ByteUtil {

	public static byte[] replace(byte[] arr, byte toFind, byte... replaceWith) {
		List<Byte> list = new LinkedList<Byte>(Arrays.asList(ArrayUtils.toObject(arr)));
		Byte find = new Byte(toFind);
		List<Byte> replace = Arrays.asList(ArrayUtils.toObject(replaceWith));
		int idx = 0;
		while ((idx = list.indexOf(find)) != -1) {
			list.remove(idx);
			list.addAll(idx, replace);
		}
		Byte[] result = new Byte[list.size()];
		list.toArray(result);
		return ArrayUtils.toPrimitive(result);
	}
	
}
