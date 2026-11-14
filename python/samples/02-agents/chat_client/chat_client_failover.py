# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import os
from typing import Annotated

from agent_framework import Agent, FailoverChatClient, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

"""
Chat Client Failover Example

This sample sends each model call to an ordered list of chat clients with `FailoverChatClient`. When a
client fails (for example with a 429, a 5xx, or a regional outage), the same model call is retried on the
next client. Failover happens per model call inside the tool loop, so tools that already ran are not run
again, and the next client sees the tool results produced so far.

To show failover without waiting for a real outage, the primary client points at a deployment name that does
not exist, so every call to it fails. In production the clients would be, for example, a deployment in a
second region or a model from another provider.

Conversation history stays on the client side: `FailoverChatClient` sends `store=False` to clients that store
history in the service by default, because history stored by one service cannot be read by another.

Environment variables:
    FOUNDRY_PROJECT_ENDPOINT - Microsoft Foundry project endpoint.
    FOUNDRY_MODEL - Model deployment name used by the fallback client, for example `gpt-4o`.
"""

refunds_issued: list[str] = []


# NOTE: approval_mode="never_require" is for sample brevity.
@tool(approval_mode="never_require")
def issue_refund(order_id: Annotated[str, Field(description="The order to refund.")]) -> str:
    """Issue a refund for an order."""
    refunds_issued.append(order_id)
    return f"Refund issued for order {order_id}."


async def main() -> None:
    # Show the warning that FailoverChatClient logs for each failover.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]

    # FailoverChatClient does not close the clients it is given, so the sample owns the credential and both
    # clients and closes them on success, failure, or cancellation.
    async with (
        AzureCliCredential() as credential,
        # 1. The primary client fails on every call because the deployment does not exist.
        FoundryChatClient(
            project_endpoint=project_endpoint,
            model="deployment-that-does-not-exist",
            credential=credential,
        ) as primary,
        # 2. The fallback client uses a working deployment.
        FoundryChatClient(
            project_endpoint=project_endpoint,
            model=os.environ["FOUNDRY_MODEL"],
            credential=credential,
        ) as fallback,
    ):
        # 3. Tools, function middleware, and agent middleware are configured once, on the failover client or agent.
        agent = Agent(
            client=FailoverChatClient([primary, fallback]),
            name="SupportAgent",
            instructions="You are a customer support agent. Use the tools you have to resolve requests.",
            tools=[issue_refund],
        )

        response = await agent.run("Please refund order 123 and confirm when it is done.")
        print(f"Agent: {response.text}")
        print(f"Refunds issued: {refunds_issued}")


if __name__ == "__main__":
    asyncio.run(main())

"""
Sample output (abridged):

WARNING Chat client 1 of 2 (FoundryChatClient) failed with ...; trying the next chat client.
Agent: I've issued a refund for order 123.
Refunds issued: ['123']
"""
