---
layout: sub_page
title: "TUI guide"
subtitle: "Update your Branch"
role: "child"
prefix: "contribute"
postfix: "tui_guide/update_your_branch"
permalink: my_collections/sub_help_pages/contribute/tui_guide/update_your_branch
---
# Update your Branch
Let's say that your branch is out of sync. Then you want to update it.
1. First checkout your branch
```
    git checkout < your branch >
```
2. The rebase with main. Be prepared for merge conflicts
```
    git rebase main
```