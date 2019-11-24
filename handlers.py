import sys

class Progbar(object):

    def __init__(self, loader):
        self.num_iterations = len(loader)
        self.output_stream = sys.stdout

    def __call__(self, engine):
        num_seen = engine.state.iteration - self.num_iterations * (engine.state.epoch - 1)

        percent_seen = 100 * float(num_seen) / self.num_iterations
        equal_to = int(percent_seen / 10)
        done = int(percent_seen) == 100

        bar = '[' + '=' * equal_to + '>' * (not done) + ' ' * (10 - equal_to) + ']'
        message = 'Epoch {epoch} | {percent_seen:.2f}% | {bar}'.format(epoch=engine.state.epoch,
                                                                       percent_seen=percent_seen,
                                                                       bar=bar)

        message += ' | {name}: {value:.2e}'.format(name='Smooth Loss', value=engine.state.metrics['smooth loss'])
        message += '\r'

        self.output_stream.write(message)
        self.output_stream.flush()

        if done:
            self.output_stream.write('\n')