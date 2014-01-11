from mrjob.job import MRJob
import mrjob

class MRWordFrequencyCount(MRJob):

    INPUT_PROTOCOL = mrjob.protocol.RawValueProtocol
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol
    HADOOP_INPUT_FORMAT = 'hadoopml.libfileinput.WholeFileInputFormat'
#     JOBCONF = { 'mapreduce.input.lineinputformat.linespermap': 2 }

    def mapper(self, _, line):
        print '\n' in line
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
#     config = RegressionConf()
#     alg = RgressionFac(config)
#     data = 'file://'
#     dataConf = NormalizeDataConf(preprocessing, )
    
    MRWordFrequencyCount.run()
