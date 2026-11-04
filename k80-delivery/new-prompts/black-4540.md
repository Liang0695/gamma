# Tokenizing many f-strings has high overhead

The report observes that tokenizing a stream of many f-string literals spends a measurable portion of its time in a generator expression. It was reported on Black main, macOS, and Python 3.12.6.

Run this minimal profiling script:

```python
import cProfile, pstats, io
from blib2to3.pgen2 import tokenize

profiler = cProfile.Profile()
example = io.StringIO(','.join(['f"X"'] * 10000)).readline
profiler.enable()
tokenize.tokenize(example, lambda *_: None)
profiler.disable()
pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME).print_stats("black", "src", 10)
```

The reported profile has 720,011 calls in 0.133 seconds; a generator expression accounts for about 15-20% of the measured time. The task is to address that observed cost without changing tokenization results.
