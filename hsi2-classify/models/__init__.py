from .CNN_HSI import CNN_HSI
from .TwoCNN import TwoCNN
from .HybridSN import HybridSN
from .RSSAN import RSSAN
from .JigSawHSI import JigSawHSI
from .resnet_HSI import resnet50


model_map = {
    'CNN_HSI': CNN_HSI,
    'TwoCNN': TwoCNN,
    'HybridSN': HybridSN,
    'RSSAN': RSSAN,
    'JigSawHSI': JigSawHSI,
    'ResNet': resnet50
}
