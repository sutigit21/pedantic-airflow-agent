from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model="gemini-2.0-flash-exp",
)

result = agent.run_sync('Who are you?')
print(result.data)