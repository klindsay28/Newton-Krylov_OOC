#!/bin/bash -i

for nbname in `ls test_run_*.ipynb`; do
    echo nbname = $nbname

    echo compare lines with datestamps, ignoring datestamps
    diff <(git diff $nbname | grep -E '^- *"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] ' | cut -f5- -d:) \
         <(git diff $nbname | grep -E '^\+ *"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] ' | cut -f5- -d:)

    echo compare lines without datestamps
    diff <(git diff $nbname | grep -E '^-' | grep -vE '^- *"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] ') \
         <(git diff $nbname | grep -E '^\+' | grep -vE '^\+ *"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] ')

done
