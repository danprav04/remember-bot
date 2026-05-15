import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db.engine import get_session_factory
from src.db.models import Fact
from sqlalchemy import select
from src.llm.embeddings import EmbeddingService
from src.config import AppConfig

async def main():
    config = AppConfig()
    embedding_service = EmbeddingService(config)
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(Fact).where(Fact.embedding.is_(None))
        result = await session.execute(stmt)
        facts = result.scalars().all()

        if not facts:
            print("No facts require embedding backfill.")
            return

        print(f"Found {len(facts)} facts to backfill. Starting...")

        for i, fact in enumerate(facts):
            try:
                print(f"Embedding fact {i+1}/{len(facts)}: {fact.content}")
                embedding = await embedding_service.embed(fact.content)
                if embedding and len(embedding) > 0:
                    fact.embedding = embedding
                    # Commit every 10 facts to avoid giant transactions
                    if i % 10 == 0:
                        await session.commit()
            except Exception as e:
                print(f"Failed to embed fact {fact.id}: {e}")

        await session.commit()
        print("Backfill complete.")

if __name__ == "__main__":
    asyncio.run(main())
