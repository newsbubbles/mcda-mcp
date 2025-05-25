# PydanticAI Agent with MCP

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import AgentRunResult

from dotenv import load_dotenv
import os
import argparse
import asyncio
import traceback
from datetime import datetime, timezone

load_dotenv()

# Set up argument parsing
parser = argparse.ArgumentParser(description="MCDA Agent with MCP Server")
parser.add_argument(
    "--model", 
    default="anthropic/claude-3.7-sonnet",
    help="Model string for OpenRouter (default: anthropic/claude-3.7-sonnet)"
)
args = parser.parse_args()

# Configure logging if LOGFIRE_API_KEY is available
if os.getenv("LOGFIRE_API_KEY"):
    import logfire
    logfire.configure(token=os.getenv("LOGFIRE_API_KEY"))
    logfire.instrument_openai()

# Set up OpenRouter based model
API_KEY = os.getenv('OPENROUTER_API_KEY')
if not API_KEY:
    print("Warning: OPENROUTER_API_KEY environment variable not set")
    print("Please set this variable or the agent will not function properly")

model = OpenAIModel(
    args.model,
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1', 
        api_key=API_KEY
    ),
)

# Environment variables for MCP server
# This agent doesn't require specific API keys but we'll forward any that are set
env = {}
# Forward any necessary environment variables
for key in os.environ:
    if key.startswith("MCDA_") or key.startswith("NUMPY_"):
        env[key] = os.getenv(key)

# Set up MCP server
mcp_servers = [
    MCPServerStdio('python', ['src/mcp_server.py'], env=env),
]

# Function to load agent prompt
def load_agent_prompt(agent: str):
    """Loads given agent replacing `time_now` var with current time"""
    print(f"Loading {agent} prompt")
    time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if agents directory exists
    agents_dir = os.path.join(os.getcwd(), "agents")
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    
    agent_path = os.path.join(agents_dir, f"{agent}.md")
    
    # Check if agent prompt file exists, create if not
    if not os.path.exists(agent_path):
        print(f"Agent prompt file not found at {agent_path}, creating default prompt")
        default_prompt = f"""# MCDA Agent

## Identity and Purpose
You are MCDAAgent, a specialized assistant for multi-criteria decision analysis. You help users evaluate alternatives against multiple criteria using various MCDA methods. The current time is {{time_now}}.

## Capabilities
- Analyze decision problems with multiple alternatives and criteria
- Apply various MCDA methods like TOPSIS, AHP, VIKOR, PROMETHEE, WSM, WPM, and WASPAS
- Calculate criteria weights using methods like AHP, entropy, and equal weighting
- Compare results across multiple MCDA methods
- Provide visualizations and sensitivity analysis

## Limitations
- You rely on the MCDA tools to perform calculations
- For complex analyses, you need complete and accurate input data
- Your recommendations are based on the mathematical models, which have their own assumptions

## Communication Style
- Be clear and concise in explaining MCDA concepts and results
- Use a professional, helpful tone
- Explain technical terms when they first appear
- Present results in an organized, easy-to-understand manner

## How to Help Users
1. Understand the decision problem and what the user wants to accomplish
2. Guide users in defining alternatives and criteria clearly
3. Help select appropriate MCDA methods based on the problem characteristics
4. Explain the results and their implications
5. Provide sensitivity analysis to test robustness of recommendations

## Example Workflows

### Basic Decision Analysis
1. Define alternatives and criteria
2. Assign weights to criteria
3. Build a decision matrix with performance values
4. Apply an MCDA method (like TOPSIS or WSM)
5. Interpret results and make recommendations

### Advanced Analysis
1. Define decision problem components
2. Calculate criteria weights using AHP or entropy
3. Compare results across multiple methods
4. Perform sensitivity analysis on weights
5. Provide comprehensive recommendations with confidence levels
"""
        with open(agent_path, "w") as f:
            f.write(default_prompt)
    
    # Load the prompt
    with open(agent_path, "r") as f:
        agent_prompt = f.read()
    
    # Replace dynamic variables
    agent_prompt = agent_prompt.replace('{time_now}', time_now)
    return agent_prompt

# Load up the agent system prompt
agent_name = "MCDAAgent"
agent_prompt = load_agent_prompt(agent_name)
print("Agent prompt loaded successfully")

# Initialize agent with MCP servers
agent = Agent(model, mcp_servers=mcp_servers, system_prompt=agent_prompt)

async def main():
    """CLI testing in a conversation with the agent"""
    print("Starting MCDA Agent with MCP Server...")
    print(f"Using model: {args.model}")
    print("Type 'exit' or 'quit' to end the conversation")
    
    async with agent.run_mcp_servers(): 
        print("MCP servers started successfully")
        result: AgentRunResult = None

        while True:
            if result:
                print(f"\n{result.output}")
            
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation. Goodbye!")
                break
                
            err = None
            for i in range(0, 3):
                try:
                    result = await agent.run(
                        user_input, 
                        message_history=None if result is None else result.all_messages()
                    )
                    break
                except Exception as e:
                    err = e
                    print(f"Attempt {i+1}/3 failed:")
                    traceback.print_exc()
                    await asyncio.sleep(2)
                    
            if result is None:
                print(f"\nError: {err}. Try again...\n")
                continue

        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
