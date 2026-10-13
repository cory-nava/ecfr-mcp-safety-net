"""Stdio smoke test for the ecfr-mcp server (free tier)."""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    env = dict(os.environ)
    env["ECFR_MCP_TIER"] = "free"
    env["ECFR_MCP_DATA"] = str(Path(__file__).parent.parent / "data")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "server.py")],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", names)
            assert {"search_corpus", "get_section", "version_history",
                    "list_programs"} <= set(names)

            r = await s.call_tool("list_programs", {})
            progs = json.loads(r.content[0].text)
            print("programs:", len(progs), "| top:", progs[0]["program"],
                  progs[0]["current_sections"])
            assert len(progs) >= 10

            r = await s.call_tool("search_corpus",
                                  {"query": "7 CFR 273.9", "top_k": 3})
            res = json.loads(r.content[0].text)
            print("cite search:", res["tier"], "->",
                  res["results"][0]["citation"])
            assert res["results"][0]["citation"] == "7 CFR 273.9"
            assert res["tier"] == "free"

            r = await s.call_tool("search_corpus",
                                  {"query": "What counts as income for SNAP?", "top_k": 3})
            res = json.loads(r.content[0].text)
            top3 = [h["citation"] for h in res["results"]]
            print("natural search top3:", top3)
            assert any("273.9" in c or "273.10" in c for c in top3), top3

            r = await s.call_tool("get_section", {"citation": "7 CFR 273.9"})
            sec = json.loads(r.content[0].text)
            print("get_section:", sec["citation"], "| chars:", len(sec["text"]))
            assert sec["found"] and "income" in sec["text"].lower()

            r = await s.call_tool("get_section",
                                  {"citation": "20 CFR 416.1102",
                                   "as_of": "2024-06-01"})
            sec = json.loads(r.content[0].text)
            print("as_of:", sec["citation"], sec["effective_from"],
                  "->", sec["effective_to"], "| current:", sec["is_current"])
            assert sec["found"]

            r = await s.call_tool("version_history",
                                  {"citation": "20 CFR 416.1102"})
            hist = json.loads(r.content[0].text)
            print("versions:", len(hist))
            assert len(hist) >= 2
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
