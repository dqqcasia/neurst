import tensorflow as tf
import logging


class Summary(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def summary_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def summary_distribution(self, distribute, name):
        """
        distribute: [time x num_output]
        """
        len_time, num_output = distribute.shape
        for time in range(len_time):
            for i in range(num_output):
                summary = tf.Summary(value=[tf.Summary.Value(tag=name+'_'+str(i),
                                                             simple_value=distribute[time][i])])
                self.writer.add_summary(summary, time)
        self.writer.flush()
        logging.info('summaried the distribution!')

    def histogram(self, tag, values, step, bins=1000):
        import numpy as np
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


if __name__ == '__main__':
    import numpy as np

    summary = Summary('/Users/easton/tmp/test')
    for step, x in enumerate(np.linspace(-5, 5, 1000)):
        summary.summary_scalar('scalar_test', np.sin(x), step)