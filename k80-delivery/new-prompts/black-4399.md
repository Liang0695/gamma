# A comment in a multiline return annotation triggers an internal error

Formatting a function whose parenthesized, multiline return annotation contains a comment can trigger an AST safety error.

Reproduce with a module that defines a class and a function shaped like this:

```python
class MyClass:
    ...

def create_my_nested_class() -> (
    # keep this annotation comment
    MyClass
):
    class MyNestedClass(MyClass):
        pass
    return MyNestedClass
```

Black should complete formatting without reporting that its output is not equivalent to the input.
