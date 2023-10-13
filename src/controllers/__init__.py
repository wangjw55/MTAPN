REGISTRY = {}

from .basic_controller import BasicMAC
from .action_basic_controller import ABasicMAC
from .n_controller import NMAC
from .action_controller import AMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["abasic_mac"] = ABasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["a_mac"] = AMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC