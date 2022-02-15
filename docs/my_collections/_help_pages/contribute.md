---
layout: page
title: "How to contribute?"
permalink: my_collections/help_pages/contribute/
---
# How to contribute?
This help page is intended for people who want to contribute to the libRAINBOW repository. 
Before you can contribute, you need to be a member of the RAINBOW team or have other 
collaborations with the IMAGE section. Otherwise please contact 
[us](https://di.ku.dk/Ansatte/forskere/?pure=da/persons/566411). 

## The Procedure
1. Clone the repository from [git](https://github.com/diku-dk/libRAINBOW/tree/main)
2. Create a [issue](https://docs.github.com/en/issues) on the [github webpage](https://github.com/diku-dk/libRAINBOW/issues)
3. Create a local branch. You will make all the changes within the local branch. 
You are **not** allowed to change the [main](https://github.com/diku-dk/libRAINBOW/tree/main) branch.
4. When all changes are done, make a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) by first pushing the branch upstream to the server. 
5. Assign the pull request to the related issue. 
6. Assign one or more reviewers.
7. Then if the reviewers ship it and all test passes, you are allowed to merge with the main branch. You must
add unittest.

## Example
Let's say you want to add another simulator, ```example_simulator.py ```, ```test_example_simulator.py``` 
and ```documentation_example_simulator.md```. ```example_simulator.py ``` is 
your simulator src code, ```test_example_simulator.py``` is the unittest file using the [unittest framework](https://docs.python.org/3/library/unittest.html), 
and ```documentation_example_simulator.md``` is the documentation page.  

### Example TUI using a [BASH terminal](https://en.wikipedia.org/wiki/Bash_(Unix_shell))
1. Clone the repository from git:\
`git clone git@github.com:diku-dk/libRAINBOW.git`
2. Use the GUI guide
3. Create a local branch:\
`git branch <issue_number>b`\
`git checkout <issue_number>b` \
Where the issue number would be **, see GUI.
4. Make pull request: \
`git push --set-upstream origin <issue_number>b` 
5. Use the GUI guide
6. Use the GUI guide
7. Use the GUI guide

### Example GUI

