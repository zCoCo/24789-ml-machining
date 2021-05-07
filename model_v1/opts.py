"""
Program options and settings, settable via command line args using argparse.

Use:
In command line to set an option:
```
python file_name.py --some-arg some_value
```

In a file which requires opts.
```
from opts import get_opts
opts = get_opts()
# ...
assert opts.some_arg == 'some_value'
```
"""

import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='24789 — ML for Machining')

    # Feature detection (requires tuning)
    parser.add_argument('--data-dir', type=str, default='../data/ForLSTM',
                        help='Where data files should be loaded from.')

    opts = parser.parse_args()

    return opts
