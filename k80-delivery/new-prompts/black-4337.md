# Multiline f-string with a multiline string in its expression cannot be parsed

On Black 24.4.2, running Black on this minimal file fails to parse the multiline f-string:

```python
f"""{'''
'''}"""
```

The report says this parsed successfully in Black versions <=24.4.0. It reproduces on Windows 11 with Python 3.10, 3.11, and 3.12. Black should parse the file successfully.
