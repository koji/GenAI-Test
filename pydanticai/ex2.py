from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str
    continent: str


agent = Agent('ollama:phi4', result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2016?')
print(result.data)
print(result.usage())
