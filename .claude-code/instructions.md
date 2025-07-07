# Claude Code Instructions for Bug Fixes

When an issue is created with the label "bug" or "test-failure":

1. Read the issue description carefully
2. Run the tests locally to reproduce the failure
3. Examine the test output to understand what's failing
4. Look at the relevant code in src/vector_ops.py
5. Fix the bug and ensure all tests pass
6. Create a pull request with the fix

## Common Commands:
- Run tests: `python tests/test_vector_ops.py`
- Install dependencies: `pip install -r requirements.txt`