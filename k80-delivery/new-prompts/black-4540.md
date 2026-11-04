# Tokenizing many f-strings has high overhead

Tokenization slows down when a file contains many f-string literals. The report identifies a large number of calls during tokenization as the observed issue.

Reproduce by tokenizing a generated stream containing 10,000 repeated `f"X"` expressions and profiling that operation. The reported profile shows substantially more work in Black's f-string tokenization path than expected.

The tokenizer should process this input without the reported performance regression while keeping its output correct.
