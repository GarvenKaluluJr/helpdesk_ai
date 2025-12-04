# scripts/load_test.py
import asyncio
import httpx

BASE_URL = "http://127.0.0.1:8000"


async def hit_dashboard(client: httpx.AsyncClient, idx: int) -> None:
    # assumes you already logged in in browser; this is just a simple GET test
    r = await client.get(f"{BASE_URL}/tickets")
    print(f"[agent {idx}] /tickets -> {r.status_code}")


async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [hit_dashboard(client, i) for i in range(10)]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
