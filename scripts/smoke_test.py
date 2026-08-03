# scripts/smoke_test.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kvd_mcp.app import get_app

def main():
    app = get_app()
    
    # 1. Run initial index and print the beautiful summary
    print("\n" + "="*60)
    print("RUNNING INITIAL INDEX...")
    print("="*60)
    summary = app.run_initial_index()
    print(f"✅ Indexed: {summary['indexed']} | Skipped: {summary['skipped']} | Deleted: {summary['deleted']} | Errors: {summary['errors']}")
    print(f"⏱️  Time: {summary['elapsed_seconds']}s\n")
    
    # 2. Test search
    test_queries = ["docker", "python", "test"]
    
    for query in test_queries:
        print(f"\n🔍 QUERY: '{query}'")
        results = app.search.search(query, limit=3)
        
        if not results:
            print("   ❌ No results found")
            continue
        
        for i, r in enumerate(results, 1):
            print(f"   [{i}] Score: {r.score:.3f} | {r.note_title}")
            print(f"       Preview: {r.text[:80]}...")

if __name__ == "__main__":
    main()