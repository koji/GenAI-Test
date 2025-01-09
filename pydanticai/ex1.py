from pydantic_ai import Agent

agent = Agent('ollama:phi4',
              system_prompt='Be concise, reply with one sentence.')
result = agent.run_sync('What is the capital of Japan and tell me about that?')
print(result.data)
