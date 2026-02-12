# Tests

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run tests with verbose output:

```bash
uv run pytest -v
```

Run a specific test file:

```bash
uv run pytest tests/electrai/model/test_srgan_layernorm_pbc.py
```

Run a specific test class:

```bash
uv run pytest tests/electrai/model/test_srgan_layernorm_pbc.py::TestGeneratorResNet
```

Run a specific test:

```bash
uv run pytest tests/electrai/model/test_srgan_layernorm_pbc.py::TestGeneratorResNet::test_generator_output_shape_default
```
