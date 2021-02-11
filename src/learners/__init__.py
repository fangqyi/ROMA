from .q_learner import QLearner
from .coma_learner import COMALearner
from .latent_q_learner import LatentQLearner
from .sc_learner import SCLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY['latent_q_learner'] =LatentQLearner
REGISTRY['sc_learner']=SCLearner
