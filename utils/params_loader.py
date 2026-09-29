"""
Utility for loading parameters from params.yaml
"""

import yaml
from pathlib import Path
from utils.logger import get_logger


logger = get_logger(__name__)


class ParamsLoader:

    def __init__(self):

        self.base_dir = Path(__file__).resolve().parent.parent
        self.params_path = self.base_dir / "params.yaml"

    def load(self):

        try:
            logger.info("Loading parameters from params.yaml")

            with open(self.params_path, "r", encoding="utf-8") as file:
                params = yaml.safe_load(file)

            logger.info("Parameters loaded successfully")

            return params

        except Exception as e:

            logger.error(
                f"Error while loading parameters: {e}"
            )

            raise