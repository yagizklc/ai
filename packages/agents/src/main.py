import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from src.config import config

"""
Site Reliability Engineering Team of Agents

they read logs and metrics of the running system and make decisions based on the data they collect
their main functionalities are:
- Summarizing the metric usage such as CPU, Memory, Disk, Network, etc.
- Detecting anomalies in the system
- Give insightful reports with respect to logs

"""

logfire.configure(service_name='sre-agents-team', token=config.logfire_token)

agent_configs = [
    (
        'LogAgent',
        'Collect logs and give insightful reports about how the users are interactig with the site',
    ),
    (
        'MetricAgent',
        'Summarize the metric usage such as CPU, Memory, Disk, Network, etc. Detect anomalies in the system if any, and notify the team',
    ),
    (
        'RoulatteAgent',
        'Use the `roulette_wheel` function to see if the customer has won based on the number they provide.',
    ),
]
agents: dict[str, Agent] = {
    name: Agent(
        model='ollama:llama3.2',
        name=name,
        system_prompt=instruction,
        model_settings=ModelSettings(
            temperature=0.0,
            max_tokens=20,
            top_p=1.0,
        ),
    )
    for name, instruction in agent_configs
}

roulette_agent = agents['RoulatteAgent']


@roulette_agent.tool  # type: ignore
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


def main():
    success_number = 18

    result = roulette_agent.run_sync(
        'Put my money on square twenty',
        deps=success_number,  # type: ignore
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
    print(result.data)


if __name__ == '__main__':
    main()
