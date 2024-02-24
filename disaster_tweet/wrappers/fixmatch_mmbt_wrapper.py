import sys
sys.path.append('../..')

from wrappers.fixmatch_disaster_wrapper import FixMatchDisasterWrapper

class FixMatchMMBTWrapper(FixMatchDisasterWrapper):
    
    def __init__(self, config):
        super(FixMatchDisasterWrapper, self).__init__(config)
