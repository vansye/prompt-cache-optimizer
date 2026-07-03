# Why Structure Optimization Improves Caching (Without Changing Output)

## First-Principles Analysis

### 1. How Prompt Caching Works

```
Request A: [tokens 1..N] → API Gateway stores hidden state for prefix
Request B: [tokens 1..M] where M ≥ threshold
           if B[1..K] == A[1..K] for K ≥ threshold:
               → reuse A's cached prefix state
               → only compute from token K+1 onwards
```

Both Anthropic and OpenAI use **exact byte-level prefix matching**.
A single changed byte in the first 1024 tokens = cache miss.

### 2. Why Structural Variations Cause Misses

Two messages with identical semantic content can differ structurally:

| Variation | Example | Effect on Cache |
|-----------|---------|----------------|
| Line endings | `\r\n` vs `\n` | Different bytes → different tokens → miss |
| JSON key order | `{"a":1,"b":2}` vs `{"b":2,"a":1}` | Different token sequence → miss |
| Whitespace | `"hello"` vs `"  hello  "` | Extra tokens → prefix shift → miss |
| Message order | System in position 0 vs 3 | Entire prefix changes → complete miss |
| Separators | `\n\n` vs `\n---\n` | Different token pattern → miss |

These are **not** semantic differences — they are encoding/formatting artifacts.
Yet they break caching because caching operates on the byte level, not the
semantic level.

### 3. Why the Transformation Is Lossless

**For JSON:**
```
json.loads(x) → json.dumps(x, sort_keys=True)

Before: {"z": 1, "a": 2}    →  token sequence: { "z" : 1 , "a" : 2 }
After:  {"a": 1, "z": 2}    →  token sequence: { "a" : 1 , "z" : 2 }

Token sequences differ, but:
- The decoded dictionary is identical: {z: 1, a: 2}
- The LLM reads the decoded values, not their position in the string
- Self-attention processes all tokens simultaneously — order within an
  object doesn't change which values the model attends to
```

**For whitespace:**
```
"hello\n \nworld" and "hello\n\nworld" produce different tokens,
but the LLM's tokenizer treats whitespace as padding, not content.
The semantic tokens ("hello", "world") and their relationship are identical.
```

**For message ordering:**
```
An LLM sees a conversation as a sequence. Moving the system message
to position 0 changes the order but not the information. The model
was trained on diverse orderings (instructions at the start, middle,
or end of documents). The *set* of content tokens is preserved.
```

### 4. The Safety Guarantee

Every stage in the optimization pipeline has a safety check:

```
Normalizer:  WhiteSpaceSanitizer tests that no content was modified
             JsonCanonicalizer preserves the original value set

Reorderer:   SafetyCheck.verify_reorder() confirms:
             - Same number of non-separator messages
             - Same (role, content) pairs (set equality)
             - System message content is preserved

Aligner:     Padding only appends to existing content
             (never replaces or truncates)
             Original content is verified as a substring of padded content

Formatter:   Strips internal markers, restructures per provider schema
             Content field values are never modified
```

If any check fails, the optimizer falls back to the original input.
**No optimization is better than a wrong optimization.**

### 5. Empirical Evidence

Anthropic's official documentation recommends:
- Place system prompts first
- Keep a consistent structure across requests
- Use `cache_control` breakpoints at stable prefix boundaries

This is exactly what this library automates.

OpenAI's caching docs similarly note:
> "Prompt Caching works best when your prompts have a long, stable prefix."

### 6. Practical Impact

In typical usage:
- **System prompt** rarely changes between requests → should be the stable prefix
- **Few-shot examples** may change → place after the system prompt
- **User input** always changes → place at the end

By ensuring the stable prefix (system + fixed instructions) is always
first and always byte-identical, the optimizer maximizes the probability
that consecutive requests share a long enough prefix for caching.
