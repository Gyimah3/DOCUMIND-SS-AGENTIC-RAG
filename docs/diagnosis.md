#!/usr/bin/env python3
"""
Redis Vector Store Diagnostic Script

This script helps diagnose why RAG retrieval returns empty results.
It checks index schema, tests searches, and verifies data integrity.

Usage:
    python diagnose_redis_vectorstore.py --index-name your_index --redis-url redis://localhost:6379
    
    # With fix flag to automatically recreate index
    python diagnose_redis_vectorstore.py --index-name your_index --fix
"""

import argparse
import asyncio
import sys
from typing import Any, Dict, List
import redis
from redis.commands.search.query import Query
import numpy as np


class RedisVectorStoreDiagnostic:
    """Diagnostic tool for Redis Vector Store issues."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = redis.Redis.from_url(redis_url)
        
    def check_connection(self) -> bool:
        """Test Redis connection."""
        try:
            self.client.ping()
            print("✅ Redis connection: OK")
            return True
        except Exception as e:
            print(f"❌ Redis connection: FAILED - {e}")
            return False
    
    def list_indexes(self) -> List[str]:
        """List all RediSearch indexes."""
        try:
            indexes = self.client.execute_command("FT._LIST")
            print(f"\n📋 Found {len(indexes)} indexes:")
            for idx in indexes:
                idx_name = idx.decode() if isinstance(idx, bytes) else idx
                print(f"   - {idx_name}")
            return [idx.decode() if isinstance(idx, bytes) else idx for idx in indexes]
        except Exception as e:
            print(f"❌ Failed to list indexes: {e}")
            return []
    
    def check_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get detailed index information."""
        try:
            info = self.client.ft(index_name).info()
            
            print(f"\n🔍 Index: {index_name}")
            print(f"   Documents: {info.get('num_docs', 0)}")
            print(f"   Index type: {info.get('index_definition', {})}")
            
            # Check for the critical bug
            attributes = info.get('attributes', [])
            has_json_paths = False
            
            print(f"\n   📊 Schema Attributes:")
            for attr in attributes:
                if isinstance(attr, list) and len(attr) >= 2:
                    identifier = attr[1].decode() if isinstance(attr[1], bytes) else attr[1]
                    
                    # Check if using JSON path syntax
                    if identifier.startswith('$.'):
                        has_json_paths = True
                        print(f"      ⚠️  {identifier} (JSON PATH SYNTAX - WRONG FOR HASH!)")
                    else:
                        print(f"      ✅ {identifier}")
            
            # Check index definition
            idx_def = info.get('index_definition', [])
            storage_type = None
            prefixes = []
            
            for i, item in enumerate(idx_def):
                if isinstance(item, bytes):
                    item = item.decode()
                
                if item == 'key_type':
                    storage_type = idx_def[i + 1].decode() if isinstance(idx_def[i + 1], bytes) else idx_def[i + 1]
                elif item == 'prefixes':
                    prefix_list = idx_def[i + 1]
                    if isinstance(prefix_list, list):
                        prefixes = [p.decode() if isinstance(p, bytes) else p for p in prefix_list]
            
            print(f"\n   📦 Storage Type: {storage_type or 'Unknown'}")
            print(f"   🏷️  Prefixes: {prefixes}")
            
            # Diagnose the issue
            if storage_type == 'HASH' and has_json_paths:
                print("\n   ⚠️  ⚠️  ⚠️  CRITICAL BUG DETECTED! ⚠️  ⚠️  ⚠️")
                print("   This HASH index uses JSON path syntax in schema!")
                print("   This is why your searches return empty results.")
                print("   FIX: Drop index and recreate with plain field names.")
            elif storage_type == 'HASH' and not has_json_paths:
                print("\n   ✅ Schema looks correct for HASH storage")
            
            return info
            
        except redis.exceptions.ResponseError as e:
            print(f"❌ Index {index_name} not found: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error checking index: {e}")
            return {}
    
    def check_documents(self, index_name: str, prefix: str = None) -> None:
        """Check if documents exist and can be retrieved."""
        
        # If no prefix provided, try to get from index info
        if not prefix:
            try:
                info = self.client.ft(index_name).info()
                idx_def = info.get('index_definition', [])
                for i, item in enumerate(idx_def):
                    if isinstance(item, bytes):
                        item = item.decode()
                    if item == 'prefixes':
                        prefix_list = idx_def[i + 1]
                        if isinstance(prefix_list, list) and len(prefix_list) > 0:
                            prefix = prefix_list[0].decode() if isinstance(prefix_list[0], bytes) else prefix_list[0]
                            break
            except:
                pass
        
        prefix = prefix or index_name
        
        print(f"\n🔎 Checking documents with prefix: {prefix}")
        
        # Check raw keys
        pattern = f"{prefix}:*"
        keys = self.client.keys(pattern)
        print(f"   Found {len(keys)} keys matching pattern '{pattern}'")
        
        if len(keys) > 0:
            # Inspect first key
            first_key = keys[0]
            if isinstance(first_key, bytes):
                first_key = first_key.decode()
            
            print(f"\n   📄 Inspecting: {first_key}")
            
            try:
                data = self.client.hgetall(first_key)
                print(f"      Fields in hash:")
                for field, value in data.items():
                    field_name = field.decode() if isinstance(field, bytes) else field
                    
                    if field_name == 'content_vector':
                        print(f"         {field_name}: <binary vector, {len(value)} bytes>")
                    elif isinstance(value, bytes):
                        try:
                            val_str = value.decode('utf-8')[:100]
                            print(f"         {field_name}: {val_str}{'...' if len(value) > 100 else ''}")
                        except:
                            print(f"         {field_name}: <binary data, {len(value)} bytes>")
                    else:
                        print(f"         {field_name}: {str(value)[:100]}")
            except Exception as e:
                print(f"      ❌ Error reading hash: {e}")
    
    def test_search(self, index_name: str) -> None:
        """Test if search returns results."""
        print(f"\n🔬 Testing search on index: {index_name}")
        
        try:
            # Simple wildcard search
            query = Query("*").return_fields("content", "filename", "type", "page_number").dialect(2)
            results = self.client.ft(index_name).search(query)
            
            print(f"   Total matches: {results.total}")
            print(f"   Documents returned: {len(results.docs)}")
            
            if len(results.docs) > 0:
                print(f"\n   📋 First result:")
                doc = results.docs[0]
                print(f"      ID: {doc.id}")
                
                # Try multiple ways to get content
                content_methods = [
                    ("getattr(doc, 'content')", getattr(doc, 'content', None)),
                    ("getattr(doc, b'content')", getattr(doc, b'content', None)),
                    ("vars(doc).get('content')", vars(doc).get('content')),
                    ("vars(doc).get(b'content')", vars(doc).get(b'content')),
                ]
                
                print(f"\n      Attempting to retrieve 'content' field:")
                content_found = False
                for method, value in content_methods:
                    if value is not None:
                        val_preview = str(value)[:100] if value else "<empty>"
                        print(f"         ✅ {method}: {val_preview}")
                        content_found = True
                    else:
                        print(f"         ❌ {method}: None")
                
                if not content_found:
                    print(f"\n      ⚠️  'content' field is EMPTY in search results!")
                    print(f"      Available attributes: {list(vars(doc).keys())}")
                    
                    # This confirms the bug
                    print(f"\n      🔴 This confirms the schema bug:")
                    print(f"         - Documents exist in Redis")
                    print(f"         - Search finds the documents")
                    print(f"         - But RETURN fields are empty")
                    print(f"         - Cause: Schema field names don't match hash keys")
                
            else:
                print(f"   ⚠️  Search found matches but returned 0 documents")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    def test_vector_search(self, index_name: str) -> None:
        """Test vector similarity search."""
        print(f"\n🎯 Testing vector search on index: {index_name}")
        
        try:
            # Create a dummy query vector (1536 dimensions for OpenAI embeddings)
            query_vector = np.random.rand(1536).astype(np.float32)
            
            query = (
                Query("(*)=>[KNN 3 @content_vector $query_vector AS vector_score]")
                .sort_by("vector_score")
                .return_fields("content", "filename", "type", "page_number")
                .dialect(2)
            )
            
            results = self.client.ft(index_name).search(
                query,
                {"query_vector": query_vector.tobytes()}
            )
            
            print(f"   Vector search results: {len(results.docs)} documents")
            
            if len(results.docs) > 0:
                doc = results.docs[0]
                content = getattr(doc, 'content', None)
                
                if content:
                    print(f"   ✅ Vector search returned content: {str(content)[:100]}...")
                else:
                    print(f"   ❌ Vector search returned empty content")
                    print(f"   Available fields: {list(vars(doc).keys())}")
            
        except Exception as e:
            print(f"   ❌ Vector search failed: {e}")
    
    def fix_index(self, index_name: str, prefix: str = None) -> bool:
        """Drop and recreate index with correct schema."""
        prefix = prefix or index_name
        
        print(f"\n🔧 FIXING INDEX: {index_name}")
        print(f"   This will drop the index and recreate it with correct schema")
        print(f"   Documents will NOT be deleted - they will be re-indexed automatically")
        
        response = input(f"\n   Proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("   Cancelled.")
            return False
        
        try:
            # Drop index (keep documents)
            print(f"\n   Dropping index {index_name}...")
            self.client.execute_command("FT.DROPINDEX", index_name)
            print(f"   ✅ Index dropped")
            
            # Recreate with correct schema
            print(f"\n   Creating index {index_name} with correct HASH schema...")
            
            from redis.commands.search.field import TextField, TagField, NumericField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
            
            schema = (
                TextField(name="content", no_stem=True),
                TagField(name="filename"),
                NumericField(name="created_at", sortable=True),
                NumericField(name="modified", sortable=True),
                TagField(name="type"),
                NumericField(name="page_number", sortable=True),
                VectorField(
                    name="content_vector",
                    algorithm="FLAT",
                    attributes={
                        "TYPE": "FLOAT32",
                        "DIM": 1536,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            
            definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
            
            self.client.ft(index_name).create_index(fields=schema, definition=definition)
            print(f"   ✅ Index created with correct schema")
            
            # Verify
            print(f"\n   Verifying fix...")
            asyncio.sleep(2)  # Give Redis time to index
            
            info = self.client.ft(index_name).info()
            num_docs = info.get('num_docs', 0)
            print(f"   ✅ Index now has {num_docs} documents")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to fix index: {e}")
            return False
    
    def run_full_diagnostic(self, index_name: str = None, fix: bool = False) -> None:
        """Run complete diagnostic."""
        print("=" * 70)
        print("REDIS VECTOR STORE DIAGNOSTIC")
        print("=" * 70)
        
        # 1. Check connection
        if not self.check_connection():
            return
        
        # 2. List indexes
        indexes = self.list_indexes()
        
        if not indexes:
            print("\n⚠️  No indexes found. Create an index first.")
            return
        
        # 3. If no index specified, use first one
        if not index_name:
            index_name = indexes[0]
            print(f"\n💡 Using index: {index_name}")
        
        # 4. Check index info
        info = self.check_index_info(index_name)
        
        if not info:
            return
        
        # 5. Check documents
        self.check_documents(index_name)
        
        # 6. Test search
        self.test_search(index_name)
        
        # 7. Test vector search
        self.test_vector_search(index_name)
        
        # 8. Fix if requested
        if fix:
            self.fix_index(index_name)
            
            # Re-test after fix
            print("\n" + "=" * 70)
            print("RE-TESTING AFTER FIX")
            print("=" * 70)
            self.test_search(index_name)
        
        print("\n" + "=" * 70)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Redis Vector Store issues")
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis connection URL"
    )
    parser.add_argument(
        "--index-name",
        help="Name of the index to diagnose (optional, will use first index if not specified)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix the index by recreating it with correct schema"
    )
    
    args = parser.parse_args()
    
    diagnostic = RedisVectorStoreDiagnostic(args.redis_url)
    diagnostic.run_full_diagnostic(args.index_name, args.fix)


if __name__ == "__main__":
    main()