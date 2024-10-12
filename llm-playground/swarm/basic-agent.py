from swarm import Swarm, Agent

client = Swarm()

english_agent = Agent(
    name="English Agent",
    model="gpt-3.5-turbo",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    model="gpt-3.5-turbo",
    instructions="You only speak Spanish.",
)

def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent

def transfer_to_english_agent():
    """Transfer english speaking users immediately."""
    return english_agent

routing_agent = Agent(
    name="Routing Agent",
    instructions="You detect the user language and pass it to correct agent.",
    functions=[transfer_to_spanish_agent, transfer_to_english_agent],
)

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
# messages = [{"role": "user", "content": "Hello. How are you?"}]
response = client.run(agent=routing_agent, messages=messages, debug=True)

print(response.messages[-1]["sender"], response.messages[-1]["content"])
