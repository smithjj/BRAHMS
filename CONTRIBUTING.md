# Contributing

For any questions, bug reports, or suggestions, please get in touch via
email: alfredo.daniel.sanchez@gmail.com

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest                  # everything
python -m pytest -m "not engine"  # skip the C++ CPU-engine build/run tests
```

The `engine` tests need g++, make, FFTW3 and HDF5 (on Windows: MSYS2 MinGW).
They build the engine in a temporary copy, so the repo is left untouched.
