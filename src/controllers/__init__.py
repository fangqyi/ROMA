REGISTRY = {}

from .basic_controller import BasicMAC
from .separate_controller import SeparateMAC
from .sc_controller import SCMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["separate_mac"]=SeparateMAC
REGISTRY["sc_mac"]=SCMAC
