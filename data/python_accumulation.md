# Accumulating successive prefixes

The `itertools.accumulate` function returns accumulated results. By default it combines items with addition.
When its input is a string, successive characters are combined into increasingly long string prefixes.
This is a general operation on strings; it does not describe any particular user text.

```text
>>> from itertools import accumulate
>>> for prefix in accumulate('abc'):
...     print(prefix)
a
ab
abc
```
