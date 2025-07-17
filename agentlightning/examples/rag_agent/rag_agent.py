import sys
import os
import time
import asyncio

from agents import (
    Agent, Runner, function_tool, set_trace_processors, gen_trace_id, trace,
    set_tracing_disabled
)
from agents.mcp import MCPServer, MCPServerSse
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
from agents.tracing.processors import ConsoleSpanExporter, BatchTraceProcessor

from agentlightning import reward, lightning_span_processor
from agentlightning.instrumentation import instrument_all

import agentops
import aiohttp

from utils import compute_scores

# TODO: Add custom code path
sys.path.append(os.path.expanduser("/scratch/amlt_code"))
print(sys.path)

AGENT_PROMPT = """You are an assistant who answers questions using Wikipedia retriever. Answer the question using only the retrieved passages. Verify your answer directly against the text.

After each search:
- Summarize findings.
- Decide if info is sufficient.
  - If sufficient: reply in <answer>...</answer> with your answer. The answer must be extremely concise: a single word or a few words only.
  - If not: suggest the next search needed to fill info gaps. The system will return top 3 relevant Wikipedia chunks.
- Explain your reasoning for the chosen action.

Repeat as needed. When done, wrap your final, concise answer in <answer> tags.
"""

async def poll_next_data_sample(root_url: str) -> dict:
    url = f"{root_url}next_data_sample"
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            except Exception as e:
                print(f"Request failed: {e}")
                await asyncio.sleep(5)
                continue

            if data.get("is_available"):
                print("Data available:", data["data"])
                return data["data"]
            else:
                print("Not available yet; retrying in 5 seconds…")
                await asyncio.sleep(5)

async def get_train_information(root_url: str) -> dict:
    url = f"{root_url}train_information"
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            except Exception as e:
                print(f"Request failed: {e}")
                await asyncio.sleep(5)
                continue
            return data

async def report_result(
    root_url: str, rollout_id: str, reward: float, trace_list: list
) -> dict:
    url = f"{root_url}report"
    payload = {
        "rollout_id": rollout_id,
        "reward": reward,
        "trace_list": trace_list
    }
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

def get_agent(model: str, openai_base_url: str, temperature: float, mcp_server) -> Agent:
    agent = Agent(
        model=LitellmModel(
            model="hosted_vllm/Qwen/Qwen3-4B", base_url=openai_base_url
        ),
        model_settings=ModelSettings(
            max_tokens=4096,
            temperature=temperature,
        ),
        name="Assistant",
        instructions=AGENT_PROMPT,
        mcp_servers=[mcp_server]
    )
    return agent

async def main() -> None:
    root_url = "http://localhost:9999/"
    openai_base_url = f"{root_url}v1"
    train_information = await get_train_information(root_url)
    print(train_information)
    model = train_information["model"]
    train_temperature = train_information["temperature"]

    while True:
        data = await poll_next_data_sample(root_url)
        rollout_id = data["rollout_id"]
        is_train = data["is_train"]
        try:
            processor = lightning_span_processor()
            with processor:
                async with MCPServerSse(
                    name="wiki_retieval_mcp",
                    params={
                        "url": "http://127.0.0.1:8099/sse",
                    },
                ) as server:
                    start_time = time.time()
                    rag_agent = get_agent(
                        model, openai_base_url,
                        train_temperature if is_train else 0.7, server
                    )
                    try:
                        question = data["question"]
                        result = await Runner.run(rag_agent, question)
                        answer = result.final_output
                    except Exception as e:
                        print("Failure:", str(e))
                        answer = "None"
                    reward_score = compute_scores(answer, str(data["answer"]))
                    end_time = time.time()
                    print("Time taken:", end_time - start_time)
                    print(f"answer: {answer} ground_truth: {data['answer']} reward: {reward_score}")
            trace_list = []
            for transition in processor.last_trace().to_trajectory():
                trace_list.append({
                    "prompt_ids": list(transition.state),
                    "response_ids": list(transition.action)
                })
            if len(trace_list) == 0:
                if is_train:
                    raise Exception("During training, every data sample must have valid runs!")
                else:
                    trace_list = [{"prompt_ids": [0], "response_ids": [0]}]
            await report_result(root_url, rollout_id, reward_score, trace_list)
        except Exception as e:
            print("Failure F:", str(e))
            await asyncio.sleep(5)

if __name__ == "__main__":
    agentops.init('xxxxxxxx')  # TODO: replace with your agentops token
    instrument_all()
    asyncio.run(main())