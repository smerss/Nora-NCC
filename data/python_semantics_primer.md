# Python sequence and output semantics

A string is a sequence of characters. Iterating over a string yields its characters in order.
The `for` statement iterates over the items of a sequence. The sequence in the example is a list, not a string.
The `print` function writes its arguments and appends a newline by default.
Calling `split()` without arguments on a string produces a sequence of words.

These are independent facts about values and operators. They do not describe a particular output layout or task.

```python
words = ["cat", "window", "defenestrate"]
for word in words:
    print(word, len(word))
```

A separate example shows a transformation from one string value to a word sequence:

```python
text = "several different words"
parts = text.split()
```
