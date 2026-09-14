# Code style

- Use descriptive `snake_case` names for variables, parameters, and configuration values. Use lowercase configuration names, including values that are treated as constants. Preserve external API names and existing data column names.
- Give each function a one-sentence docstring describing its operation.
- Put a short imperative comment above each logical block, including return blocks. Describe the operation. Keep design rationale and experimental findings in `METHODS.md`.
- Keep one statement per line. Split long expressions into named intermediate results. Format multi-argument calls on separate lines.
- Use keyword arguments for project functions and library parameters that accept them. Retain positional arguments where an interface is positional-only or accepts variable positional arguments, such as `range`, `Path`, `list.append`, `print`, and argparse option names. External parameter names such as NumPy's `a` and a model's `x` are API names, not project variable names.
- Keep function signatures to seven parameters or fewer. Separate input validation, computation, and output writing into focused helpers when an operation grows.
- Keep configuration in the command entry point, configuration object, or a clearly marked settings block. Notebook configuration belongs in its top configuration cell.
- Use normal imports. Import optional dependencies in the function that needs them. Do not load project logic through dynamic execution.
- Format every printed message as an f-string.
- Validate inputs and raise `ValueError` with a useful message.
- Specify NumPy dtypes when constructing numeric input arrays.
- Executable entry points define `main()` and use an `if __name__ == "__main__"` guard. Library modules remain importable without running jobs.

The CSV columns `x` and `y`, the saved dataset key `y`, and third-party names such as `PPM` or `IMAGENET1K_V1` are existing formats and remain unchanged.
