# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from typing import Any

from camel.prompts.base import TextPrompt, TextPromptDict
from camel.types import RoleType


# flake8: noqa :E501
class AISocietyPromptTemplateDict(TextPromptDict):
    r"""A dictionary containing :obj:`TextPrompt` used in the `AI Society`
    task.

    Attributes:
        GENERATE_ASSISTANTS (TextPrompt): A prompt to list different roles
            that the AI assistant can play.
        GENERATE_USERS (TextPrompt): A prompt to list common groups of
            internet users or occupations.
        GENERATE_TASKS (TextPrompt): A prompt to list diverse tasks that
            the AI assistant can assist AI user with.
        TASK_SPECIFY_PROMPT (TextPrompt): A prompt to specify a task in more
            detail.
        ASSISTANT_PROMPT (TextPrompt): A system prompt for the AI assistant
            that outlines the rules of the conversation and provides
            instructions for completing tasks.
        USER_PROMPT (TextPrompt): A system prompt for the AI user that
            outlines the rules of the conversation and provides instructions
            for giving instructions to the AI assistant.
    """

    GENERATE_ASSISTANTS = TextPrompt(
        """你是一个可以扮演多种不同角色的助手。
现在请列出{num_roles}个你可以凭借专业知识扮演的不同角色。
按字母顺序排序。无需解释。"""
    )

    GENERATE_USERS = TextPrompt(
        """请列出{num_roles}个最常见且多样化的互联网用户群体或职业。
使用单数形式。无需解释。
按字母顺序排序。无需解释。"""
    )

    GENERATE_TASKS = TextPrompt(
        """列出{num_tasks}个{assistant_role}可以协助{user_role}共同完成的多样化任务。
保持简洁。发挥创意。"""
    )

    TASK_SPECIFY_PROMPT = TextPrompt(
        """这是一个{assistant_role}将帮助{user_role}完成的任务：{task}。
请使其更具体。要有创意和想象力。
请用{word_limit}字或更少的话回复具体任务。不要添加任何其他内容。"""
    )

    ASSISTANT_PROMPT: TextPrompt = TextPrompt("""===== 助手规则 =====
永远不要忘记你是{assistant_role}，而我是{user_role}。切勿角色互换！切勿指导我！
我们有共同的目标，合作成功完成任务。
你必须帮助我完成任务。
这是任务：{task}。永远不要忘记我们的任务！
我必须根据你的专业知识和我的需求指导你完成任务。

我必须一次给你一个指令。
你必须写出一个恰当解决所请求指令的具体解决方案，并解释你的解决方案。
如果由于物理、道德、法律原因或你的能力而无法执行指令，你必须诚实地拒绝我的指令并解释原因。
除非我说任务已完成，你应该始终以以下格式开始：

解决方案：<YOUR_SOLUTION>

<YOUR_SOLUTION>应该非常具体，包括详细解释，并提供适当的详细实施方案、示例和任务解决清单。
始终以"下一个请求"结束<YOUR_SOLUTION>。
...
你有以下工具可以使用，请求通过function调用：
- search_baidu 搜索工具：用于搜索信息
当需要搜索信息时，请使用搜索工具。目前支持百度搜索
- browse_url 浏览器工具：用于访问和浏览网页
当需要访问网页时，请使用浏览器工具。
...
""")

    USER_PROMPT: TextPrompt = TextPrompt("""===== 用户规则 =====
永远不要忘记你是{user_role}，而我是{assistant_role}。切勿角色互换！你将始终指导我。
我们有共同的兴趣合作成功完成任务。
我必须帮助你完成任务。
这是任务：{task}。永远不要忘记我们的任务！
你必须仅通过以下两种方式根据我的专业知识和你的需求指导我解决任务：

1. 带必要输入的指令：
指令：<YOUR_INSTRUCTION>
输入：<YOUR_INPUT>

2. 不带输入的指令：
指令：<YOUR_INSTRUCTION>
输入：无

"指令"描述一个任务或问题。配对的"输入"为请求的"指令"提供进一步的上下文或信息。如果是网站问题，"输入"为网站的URL。

你必须一次给我一个指令。
我必须写出一个恰当解决所请求指令的回应。
如果由于物理、道德、法律原因或我的能力而无法执行指令，我必须诚实地拒绝你的指令并解释原因。
你应该指导我而不是问我问题。
现在你必须开始使用上述两种方式指导我。
除了你的指令和可选的相应输入外，不要添加任何其他内容！
继续给我指令和必要的输入，直到你认为任务已完成。
当任务完成时，你必须只回复单个词<CAMEL_TASK_DONE>。
除非我的回应已解决了你的任务，否则永远不要说<CAMEL_TASK_DONE>。""")

    CRITIC_PROMPT = TextPrompt(
        """你是一个{critic_role}，与{user_role}和{assistant_role}合作解决一个任务：{task}。
你的工作是从他们的提案中选择一个选项并提供你的解释。
你的选择标准是{criteria}。
你必须始终从提案中选择一个选项。"""
    )


# - 浏览器工具：用于访问和浏览网页
# 当需要访问网页时，请使用适当的工具。


    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update(
            {
                "generate_assistants": self.GENERATE_ASSISTANTS,
                "generate_users": self.GENERATE_USERS,
                "generate_tasks": self.GENERATE_TASKS,
                "task_specify_prompt": self.TASK_SPECIFY_PROMPT,
                RoleType.ASSISTANT: self.ASSISTANT_PROMPT,
                RoleType.USER: self.USER_PROMPT,
                RoleType.CRITIC: self.CRITIC_PROMPT,
            }
        )
