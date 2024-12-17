from abc import abstractmethod, ABC


class Coffee(ABC):
    @abstractmethod
    def serveCoffee(self):
        pass
        
class EspressoCoffee(Coffee):
    def serveCoffee(self):
        return "this is esspresso senior"
    
class CappucinoCoffee(Coffee):
    def serveCoffee(self):
        return "cappucino cappucino baybay"

class CoffeeFactory:
    def prepapretheCoffe(self, coffetype: str) -> str :
        if coffetype == "Espresso":
            return EspressoCoffee().serveCoffee()
        elif coffetype == "Cappucino":
            return CappucinoCoffee().serveCoffee()
        else :
            return "Don't know"
        
if __name__ == "__main__":
    coffeeType = "Cappucino"
    
    coffeeMachine =  CoffeeFactory()
    out = coffeeMachine.prepapretheCoffe(coffetype= coffeeType)
    print(out)

