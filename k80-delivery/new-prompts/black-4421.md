# A valid raw f-string containing an escaped brace cannot be parsed

Black fails to format a valid Python 3.12 raw f-string that contains an escaped opening brace adjacent to an interpolation.

Reproduce with this line:

```python
rf'{a}\{{[\}}'
```

The report uses Black 24.4.3.dev27 with CPython 3.12.4 and observes a parse error. Black should parse and format the valid f-string.
