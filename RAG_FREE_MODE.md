"""
FREE MODE RAG GUIDE
===================

This guide explains the FREE MODE feature for the RAG (Retrieval-Augmented 
Generation) system in the AI Compliance Agent.

What is FREE MODE?
------------------

FREE MODE is a keyword-based fallback mode for the RAG system that eliminates
the dependency on OpenAI's Embeddings API. This allows the compliance system
to work end-to-end without requiring an embeddings API quota.

Key Benefits:
- No external embedding API calls
- Zero cost for policy retrieval
- Works offline or in development environments
- Automatic fallback when API key is unavailable
- Same interface as vector-based retrieval


How FREE MODE Works
-------------------

Instead of computing dense vector embeddings, FREE MODE uses simple keyword
overlap scoring to retrieve relevant policies:

1. Query Keywords Extraction
   - Extract all words from the query (transcript)
   - Remove stop words (the, and, a, etc.)
   - Filter by minimum length (3+ characters)

2. Policy Keyword Extraction
   - Same process for each policy chunk
   - Creates a set of keywords for comparison

3. Similarity Scoring
   - Count overlapping keywords between query and chunk
   - Score = overlap_count / query_keyword_count
   - Range: [0.0, 1.0]

4. Results
   - Return top_k chunks sorted by score (descending)
   - Same format as vector-based retrieval
   - Maintains domain filtering

Example Scoring:

Query: "We guarantee 15% returns with zero risk"
Keywords: {guarantee, returns, zero, risk}

Policy 1: "Guaranteed returns are prohibited by law"
Keywords: {guaranteed, returns, prohibited, law}
Overlap: {returns} → Score = 1/4 = 0.25

Policy 2: "Risk disclosure requirements for investments"
Keywords: {risk, disclosure, requirements, investments}
Overlap: {risk} → Score = 1/4 = 0.25


Enabling FREE MODE Automatically
---------------------------------

FREE MODE is enabled automatically in these cases:

1. No OpenAI API Key Set
   - If OPENAI_API_KEY environment variable is empty or not set
   - Fallback happens gracefully without errors

2. Explicit Environment Variable
   - Set COMPLIANCE_FREE_MODE=true
   - Forces keyword-based retrieval regardless of API key

3. Missing openai Library
   - If the OpenAI Python library is not installed
   - System falls back to FREE MODE automatically


Using FREE MODE Explicitly
---------------------------

In code:

    from compliance_engine.rag.embeddings import EmbeddingsClient
    from compliance_engine.rag.retriever import PolicyRetriever
    from compliance_engine.compliance_engine import ComplianceEngine
    
    # Create embeddings client in FREE MODE
    embeddings_client = EmbeddingsClient(use_free_mode=True)
    
    # Create retriever with FREE MODE embeddings
    retriever = PolicyRetriever(chunks, embeddings_client=embeddings_client)
    
    # Create compliance engine
    engine = ComplianceEngine(retriever=retriever)
    
    # Use normally - no API calls will be made for embeddings
    result = engine.analyse(
        transcript="We guarantee 15% returns",
        domain="fintech"
    )


From Command Line:

    # Enable FREE MODE
    export COMPLIANCE_FREE_MODE=true
    
    # Or just don't set OPENAI_API_KEY and it auto-enables
    unset OPENAI_API_KEY
    
    # Run compliance check
    python cli.py -t "We guarantee 15% returns" --domain fintech


API Changes Summary
-------------------

EmbeddingsClient:
- New parameter: use_free_mode (bool | None)
- Auto-detection based on API key or COMPLIANCE_FREE_MODE env var
- embed_text() returns [0.0] in FREE MODE (no API call)
- No breaking changes to existing code

PolicyRetriever:
- _build_index() is now a no-op in FREE MODE
- retrieve() uses keyword matching in FREE MODE
- Same return format: List[Tuple[PolicyChunk, float]]
- Domain filtering works identically

ComplianceEngine:
- No changes to public API
- Works seamlessly with FREE MODE retriever
- RAG context is included in LLM prompt (from keyword retrieval)


Performance Characteristics
----------------------------

FREE MODE Performance:
- Policy Loading: O(n) where n = number of chunks
- Retrieval: O(n * m) where n = chunks, m = avg keywords per chunk  
- Typical Query Time: <10ms for 100 chunks
- No network I/O, fully local computation

Vector-Based Performance (with API):
- Policy Loading: O(n * k) where k = API call latency (~100-200ms)
- Retrieval: O(n) with pre-computed vectors (fast)
- Typical Query Time: ~100-200ms (API call) + <1ms (similarity)


Accuracy Comparison
-------------------

Vector Embeddings (Normal Mode):
- Semantic understanding of content
- Handles synonyms and conceptually similar text
- Better for complex regulatory language
- Requires API quota and network access
- Cost: $0.02 per 1M input tokens (text-embedding-3-small)

Keyword Matching (FREE MODE):
- Exact word matching
- Better control, no "magic"
- Suitable for compliance rules with specific terminology
- Zero cost, works offline
- Limitations: Doesn't handle synonyms


Best Practices
--------------

1. Policy Writing
   - Use consistent terminology across policies
   - Avoid too many synonyms in the same concept
   - Include relevant keywords in policy titles/headers
   - Example good: "Guaranteed returns are prohibited"
   - Example poor: "Investment guarantees are not allowed"

2. Query/Transcript Quality
   - Include relevant regulatory keywords
   - Specify the domain clearly
   - Example good: "We guarantee 15% annual returns"
   - Example poor: "The performance will be good"

3. Domain Filtering
   - Always specify domain for accurate filtering
   - Ensures only relevant policies are considered
   - Examples: "fintech", "insurance", "healthcare"

4. Fallback Strategy
   - Use FREE MODE for development/testing
   - Switch to vector embeddings for production if needed
   - Monitor retrieval quality and adjust policies as needed


Testing FREE MODE
-----------------

Run the FREE MODE test suite:

    pytest -v tests/test_rag_free_mode.py

This validates:
- FREE MODE auto-detection
- Keyword extraction and filtering
- Similarity scoring
- Retrieval ranking
- Domain filtering
- ComplianceEngine integration


Troubleshooting
---------------

Issue: "No policies retrieved"
- Check if policies exist in COMPLIANCE_POLICIES_DIR
- Verify domain filtering matches policy domain
- Check keyword overlap manually

Issue: "Wrong policies retrieved"
- Review keyword extraction (check what words are extracted)
- Verify policy terminology matches query terminology
- Consider adding more specific keywords to policies

Issue: "API calls still being made"
- Check that COMPLIANCE_FREE_MODE=true is set
- Verify OPENAI_API_KEY is not in environment
- Check that EmbeddingsClient was initialized with use_free_mode=True

Issue: "System slower than expected"
- For large policy sets (>1000 chunks), consider:
  - Splitting policies into smaller chunks
  - Using more specific domain filtering
  - Preparing pre-indexed keyword data


Migrating to Vector Embeddings
-------------------------------

If you need better accuracy later, migrating to vector embeddings is simple:

1. Set OPENAI_API_KEY environment variable
2. Ensure openai package is installed
3. System will automatically use vector embeddings
4. No code changes needed!

The EmbeddingsClient and PolicyRetriever handle the switch automatically.


Contributing & Feedback
----------------------

FREE MODE is designed to be:
- Extensible: Easy to swap retrieval strategies
- Testable: Comprehensive test coverage
- Maintainable: Clear, well-documented code

Suggestions for improvement:
- Better keyword filtering (TF-IDF, BM25)
- Chunk pre-indexing for faster retrieval  
- Hierarchical domain filtering
- Custom stop word lists per domain

Report issues or suggestions in the project repository.
"""