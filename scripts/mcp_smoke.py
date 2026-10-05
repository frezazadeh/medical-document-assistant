"""Connects to the MCP server the way a real client would (stdio) and uses it.

    python scripts/mcp_smoke.py

Handy for checking the server without setting up Claude Desktop or an IDE.
It reads whatever is in the data directory, so upload a document first.
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"])
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        print("tools:    ", [t.name for t in tools.tools])
        templates = await session.list_resource_templates()
        print("resources:", [t.uri_template for t in templates.resource_templates])
        prompts = await session.list_prompts()
        print("prompts:  ", [p.name for p in prompts.prompts])

        documents = await session.call_tool("list_documents", {})
        print("\nlist_documents ->", text_of(documents)[:300])

        hits = await session.call_tool(
            "search_documents", {"query": "hyperkalemia in the 10 mg group", "top_k": 2}
        )
        print("\nsearch_documents ->", text_of(hits)[:500])

        first_id = json.loads(text_of(documents))["result"][0]["id"]
        resource = await session.read_resource(f"document://{first_id}")
        print(f"\ndocument://{first_id} ->", resource.contents[0].text[:200])

        prompt = await session.get_prompt("grounded_clinical_qa", {"question": "Was it safe?"})
        print("\ngrounded_clinical_qa ->", prompt.messages[0].content.text[:160])


def text_of(result) -> str:
    if result.structured_content:
        return json.dumps(result.structured_content, ensure_ascii=False)
    return " ".join(block.text for block in result.content if block.type == "text")


if __name__ == "__main__":
    asyncio.run(main())
