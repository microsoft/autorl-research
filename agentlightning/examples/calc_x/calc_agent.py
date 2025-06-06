import math
import string
import re
from typing import Any

from agentlightning.client import SamplingParameters
import sympy
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

from agentlightning import Trainer, LitAgent, SamplingParameters, reward, configure_logger

configure_logger()

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])


# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py


def normalize_option(option: str) -> str:
    """
    >>> normalize_option("  (A)  \n")
    'A'
    """
    return re.sub(r"(\s+|\(|\))", "", option)


def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


def float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    pred_result = str(pred_result) if pred_result is not None else ""
    true_result = str(true_result) if true_result is not None else ""

    if pred_result.strip() == true_result.strip():
        return True

    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result

    # The task is to calculate the result as a number
    try:
        pred_float = float_eval(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:
        pass

    return False


@reward
async def eval(prediction: str, ground_truth: str) -> float:
    return float(scalar_are_results_same(prediction, ground_truth, 1e-2))


def get_agent(model, openai_base_url, temperature, workbench):
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url,
        api_key="token-abc123",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=temperature,
    )

    calc_agent = AssistantAgent(
        name="calc",
        model_client=model_client,
        workbench=workbench,
        reflect_on_tool_use=True,
    )
    return calc_agent


class CalcAgent(LitAgent):

    async def training_rollout_async(
        self, sample: Any, *, sampling_parameters: SamplingParameters | None = None, rollout_id: str | None = None
    ) -> Any:
        assert sampling_parameters is not None
        async with McpWorkbench(calculator_mcp_server) as workbench:
            calc_agent = get_agent(
                sampling_parameters["model"],
                self.trainer.get_openai_endpoint(),
                sampling_parameters["temperature"],
                workbench,
            )
            try:
                output_format = "Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: <answer> ###."
                task = sample["question"] + " " + output_format
                result = await calc_agent.run(task=task)
                # evaluate
                answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", result.messages[-1].content)
                if answer:
                    answer = answer.group(1)
                else:
                    answer = result.messages[-1].content
            except Exception as e:
                print("Failure:", str(e))
                answer = "None"
            reward = await eval(answer, str(sample["result"]))  # reward is tracked with the decorator
            print("answer: {} ground_truth: {} reward: {}".format(answer, sample["result"], reward))

    async def validation_rollout_async(
        self, sample: Any, *, sampling_parameters: SamplingParameters | None = None, rollout_id: str | None = None
    ) -> Any:
        return await self.training_rollout_async(
            sample, sampling_parameters={"temperature": 0, "model": sampling_parameters["model"]}, rollout_id=rollout_id
        )


if __name__ == "__main__":
    Trainer(n_workers=4).fit(CalcAgent(), "http://localhost:9999/")
