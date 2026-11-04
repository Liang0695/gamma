# Parenthesized context manager is rejected during formatting

With Python 3.12 syntax enabled, Black fails while formatting a parenthesized context manager containing the same name twice.

Reproduce with:

```python
with (name_2, name_2) as name_2:
    pass
```

Run `black -l 100 -t py312 bug.py`. Black reports an internal error or that its output is not equivalent to the source. It should format this valid statement without changing its meaning or failing.
