# Triple-quoted string causes an internal formatting error

Black reports that its output is not equivalent to the source when formatting a triple-quoted string with unusual indentation and trailing blank space inside the string.

Reproduce with:

```python
if __name__ == "__main__":
    """     This string...
    ... will confuse black.
     
    """
    pass
```

Run `black file.py`. Black should preserve the string's meaning and finish without an internal error.
