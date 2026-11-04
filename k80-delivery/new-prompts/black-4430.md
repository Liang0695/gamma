# Line-range formatting changes code outside the requested range

Black's `--line-ranges` option is expected to limit formatting to the requested lines, but formatting a selected section also changes whitespace and code outside that section.

Reproduce with a file containing two separated blocks of Python statements, each surrounded by deliberately irregular comment indentation. Run:

```sh
black --line-ranges 1-9 file.py
```

The selected first block may be formatted, but the later block outside lines 1–9 should remain byte-for-byte unchanged. The reported behavior changes that later block too.
