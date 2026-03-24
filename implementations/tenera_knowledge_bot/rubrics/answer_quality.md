You must emit exactly the following metrics and no others:

1. factual_correctness
   - Value must be 1 only if the candidate answer is materially consistent with the expected output.
   - Otherwise value must be 0.
2. database_grounding
   - Value must be 1 only if the candidate answer appears to stay grounded in information that plausibly comes from the database-backed task.
   - Otherwise value must be 0.
3. completeness
   - Value must be 1 only if the candidate answer includes the important information needed to answer the input.
   - Otherwise value must be 0.
4. caveat_quality
   - Value must be 1 if the candidate handles uncertainty or missing data appropriately when needed.
   - If no caveat is needed and the answer is still appropriately scoped, value must be 1.
   - Otherwise value must be 0.

For each metric:
- Use exactly the metric names above.
- Use binary values only (0 or 1).
- Include a one-sentence metric comment.
