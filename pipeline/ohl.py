from .base import BasicPipeline


class OHLPipeline(BasicPipeline):
    
    def __init__(self, dist, common_kwargs, epochs):
        super(OHLPipeline, self).__init__(dist, common_kwargs)
        self.epochs = epochs
    
    def search(self):
        pass
    
    def test(self):
        pass
    
    def finalize(self):
        pass
