---
layout: sub_page
title: "TUI guide"
subtitle: "Upload Code"
role: "parent"
prefix: "contribute"
postfix: "tui_guide/update_code"
permalink: my_collections/sub_help_pages/contribute/tui_guide/update_code
---

# Upload Code
1. Clone the repository from git:\
`git clone git@github.com:diku-dk/libRAINBOW.git`
2. Use the GUI guide
3. Create a local branch:\
`git branch <issue_number>b`\
`git checkout <issue_number>b`\
Where the issue number would be **, see GUI.
4. Insert you code by
    1. By making a folder in isl
        ```
            cd python/isl
            mkdir < your code name >
        ```
    2. Enter your folder
        ```
            cd < your code name >
        ```
    3. The copy all your code into here 
5. Adding test
    1. In isl, find test
    ```
     cd ..
     cd test
    ```
    2. creat a test folder
    ```
        mkdir test_< your code name >
    ```
    3. Enter your test folder and make a test file
    ```
        cd test_< your code name >
        touch test_< your code name>.py
    ```
    4. Copy paste this template into you test folder

    ```python
    import unittest
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../../")
    import isl.my_folder.my.api as api


    class TestMyAPI(unittest.TestCase):

        def test_generate_unique_name_1(self):
            expected = [...]
            actual   = [...]
            self.assertNotEqual(actual, expected)
    ```

4. Make pull request: \
`git push --set-upstream origin <issue_number>b` 
5. Use the GUI guide for the rest