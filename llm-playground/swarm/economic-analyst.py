from typing import List
from swarm import Swarm, Agent
import pandas as pd
import numpy as np

client = Swarm()

# Define the functions and tools available for the agents

def get_data_for_country(country: str) -> List[float]:
    """Get the economic data for a specific country code."""
    data = pd.DataFrame({
        "Year": np.arange(2010, 2020),
        "CPI": np.random.uniform(1, 3, 10),
        "Unemployment": np.random.uniform(4, 6, 10),
        "GDP": np.random.uniform(1, 3, 10),
    })
    data = [1.5, 3.2]
    if country == "US":
        data = [2.5, 38.2]
    if country in ["GB", "UK"]:
        data = [-3.5, -42.2]
    
    return data

def calculate_inflation_rate(data: List[float]) -> float:
    """Calculate the inflation rate in percentage."""
    # inflation_rate = data["CPI"].pct_change().mean() * 100
    inflation_rate = data[1]
    return inflation_rate


def calculate_unemployment_rate(data: List[float]) -> float:
    """Calculate the unemployment rate in percentage."""
    # unemployment_rate = data["Unemployment"].mean()
    unemployment_rate = data[1]
    return unemployment_rate


def calculate_gdp_growth_rate(data: List[float]) -> float:
    """Calculate the GDP growth rate in percentage."""
    # gdp_growth_rate = data["GDP"].pct_change().mean() * 100
    gdp_growth_rate = data[1]
    return gdp_growth_rate

def get_weather_data(city: str) -> List[float]:
    """Get the weather data for a specific city."""
    return np.random.uniform(10, 30, 10).tolist()

economic_analyst = Agent(
    name="Economic Analyst",
    model="gpt-3.5-turbo",
    instructions="You are an helpful economic data analyst.",
    functions=[get_data_for_country, calculate_inflation_rate, calculate_unemployment_rate, calculate_gdp_growth_rate],
)

def transfer_to_economic_analyst():
    """Transfer users to the economic analyst."""
    return economic_analyst

assistant = Agent(
    name="Assistant",
    model="gpt-3.5-turbo",
    instructions="You are an assistant to help the user.",
    functions=[transfer_to_economic_analyst, get_weather_data],
)


response = client.run(
   agent=assistant,
   messages=[{"role":"user", "content": "What is the economic situation in england?"}],
   context_variables={"user_name":"John"},
   debug=True
)
print(response.messages[-1]["content"])