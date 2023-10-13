REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .action_rnn_agent import ARNNAgent
from .perceive_agent import PerceiveAgent
from .decision_agent import DecisionAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["a_rnn"] = ARNNAgent
REGISTRY["perceive"] = PerceiveAgent
REGISTRY["decision"] = DecisionAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent