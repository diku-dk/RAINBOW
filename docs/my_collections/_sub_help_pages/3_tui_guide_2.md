---
layout: sub_page
title: "TUI guide"
subtitle: "Squeeze Commits"
role: "child"
prefix: "contribute"
postfix: "tui_guide/squeeze_commits"
permalink: my_collections/sub_help_pages/contribute/tui_guide/squeeze_commits
---
# Squeeze Commits
This tutorial assumes that you work on your private branch. In other words, 
you are not allowed to squeeze commits.

1. Checkout your branch 
    ```
        git checkout < your branch > 
    ```

2. Assume you want to squeeze 3 commits:
    ```
        git log
    ```
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/squeeze/commits3.png)

3. The type
    ```
        git rebase -i HEAD~3
    ``` 
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/squeeze/squeeze_1.png)

4. To squeeze commit 2 and 3, type f in front of them
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/squeeze/squeeze_2.png)

5. Check that it worked using git log
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/squeeze/squeeze_3.png)