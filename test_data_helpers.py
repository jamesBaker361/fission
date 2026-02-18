from data_helpers import *
import unittest

class DataTest(unittest.TestCase):
    def test_3d(self):
        for dataset_class in [ShapeNetImageData,ShapeNetImageDataPaired]:
            with self.subTest(dataset_class=dataset_class):
                data=dataset_class("shapenet_renders",dim=(64,64),limit=10) #shapenet renders is the big directory
                print(dataset_class,"len",len(data))
                for batch in data:
                    break
                
    def test_vanilla(self):
        for dataset_class in [TextImageWikiData,VirtualTryOnData]:
            with self.subTest(dataset_class=dataset_class):
                data=dataset_class("test",dim=(64,64),limit=10)
                print(dataset_class,"len",len(data))
                for batch in data:
                    break
                
    def test_laion(self):
        data=LaionDataset((64,64))
        count=0
        for batch in data:
            pass
                
if __name__=="__main__":
    unittest.main()