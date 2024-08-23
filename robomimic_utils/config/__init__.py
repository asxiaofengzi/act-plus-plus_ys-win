from robomimic_utils.config.config import Config
from robomimic_utils.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from robomimic_utils.config.bc_config import BCConfig
from robomimic_utils.config.bcq_config import BCQConfig
from robomimic_utils.config.cql_config import CQLConfig
from robomimic_utils.config.iql_config import IQLConfig
from robomimic_utils.config.gl_config import GLConfig
from robomimic_utils.config.hbc_config import HBCConfig
from robomimic_utils.config.iris_config import IRISConfig
from robomimic_utils.config.td3_bc_config import TD3_BCConfig
from robomimic_utils.config.diffusion_policy_config import DiffusionPolicyConfig
from robomimic_utils.config.act_config import ACTConfig
