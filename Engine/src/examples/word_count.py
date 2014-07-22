from mrjob.job import MRJob
import mrjob
import sys
from protocol.NLineInputProtocol import NLineInputProtocol

class MRWordCount(MRJob):

    INPUT_PROTOCOL = NLineInputProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.NLineFileInputFormat'
    JOBCONF = { 'hadoopml.fileinput.linespermap': 2 }

    def mapper(self, key, lines):
        sys.stderr.write('  key: "' + str(key) + '"\n')
        sys.stderr.write('lines: "'+ str(lines) + '"\n')
        chars = 0
        words = 0
        for line in lines:
            sys.stderr.write('line: "' + line + '"\n')
            chars += len(line)
            words += len(line.split())
            sys.stderr.write('chars: "'+ str(chars) + '"\n')
            sys.stderr.write('words: "'+ str(words) + '"\n')
        yield "chars", chars
        yield "words", words
        yield "lines", len(lines)

    def reducer(self, key, values):
        yield key, sum(values)

    def steps(self):
        return [ self.mr( mapper = self.mapper, reducer = self.reducer ) ]


if __name__ == '__main__':
    MRWordCount().run()
