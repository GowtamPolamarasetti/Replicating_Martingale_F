from abc import ABC, abstractmethod
from src.products.base_product import BaseProduct

class BaseModel(ABC):
    """
    Abstract base class for all pricing and risk models.
    
    This ensures that any model we create will have a consistent interface,
    making them interchangeable in our experiment scripts.
    """
    def __init__(self, config: dict, product: BaseProduct):
        """
        Initializes the model with its configuration and the financial product it will price.
        
        Args:
            config (dict): The experiment configuration.
            product (BaseProduct): An instance of a financial product class.
        """
        self.config = config
        self.product = product

    @abstractmethod
    def run(self, **kwargs):
        """
        The main execution method for the model.
        
        This abstract method must be implemented by all child classes. It can be
        used for pricing at time 0, calculating a risk distribution, or any
        other primary function of the model.
        """
        pass