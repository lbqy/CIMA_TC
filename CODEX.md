# Code Quality
1. Write clean, concise, readable, extensible code with a consistent style. Avoid hidden fallback logic, silent error handling, and vague exceptions. Classify invalid or exceptional cases clearly, and raise errors or warnings with actionable messages.

2. When modifying code, do not simply patch around the symptom. First understand the root logic and data flow, then fix the underlying issue while keeping the design coherent and simple. The goal is code quality, not minimal diff size.

3. Use English for all comments, docstrings, error messages, and test names. Add clear docstrings for public functions and important helpers, explaining parameter meanings, data types, return values, and possible exceptions. Use type annotations whenever practical.

4. Write sufficient pytest tests for each feature or bug fix, covering normal cases, edge cases, and failure cases. Provide simple usage examples when helpful.

5. After completing and verifying a coherent module or milestone, manage progress with Git. Commit meaningful changes with a clear message, and report what changed, why, which tests were run, and any known limitations.

# Environments
