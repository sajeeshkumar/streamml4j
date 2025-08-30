# Contributing to streamml4j

🚀 Thanks for your interest in contributing!

## Development workflow
1. Fork the repository and clone your fork.
2. Create a new branch off `main` for your change.
3. Run `mvn clean verify` to ensure all tests pass.
4. Push your branch and open a Pull Request.

## Pull Request rules
- All changes must come through a PR.
- At least 1 maintainer review is required before merging.
- CI checks (Maven build/tests) must pass.
- Squash & merge is preferred to keep history clean.

## Code style
- Java 17, Maven.
- Keep modules small and cohesive (`core`, `algorithms`, `metrics`, `examples`).
- Add unit tests for new algorithms.
