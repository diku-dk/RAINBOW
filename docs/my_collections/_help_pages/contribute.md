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
First download [git cola](https://git-cola.github.io/):
1. Clone using git cola ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_clone.png)
2. Go to the github issue [webpage](https://github.com/diku-dk/libRAINBOW/issues)
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_new_issue.png) 
    1. Add the issue name
    2. Write a short description of what you are training to solveWrite a short 
    3. Add assignee(s)
    4. Add label
    5. Submit issue
    6. Finished issue
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_create_new_issue.png)
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/finish_issue.png)
3. Create a local branch using git cola
    1. Click on 'Branch'
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_cola_make_branch_1.png)
    2. Click on 'Create'
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_cola_make_branch_2.png)
    3. Initialize branch 
        1. Naming your branch is up to you. However, we recommend including the issue number in the name eg.
           '15b'  
        2. Make sure you branch from the main branch. If not 'checkout' the main branch first. 
        3. Create branch 
        ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_cola_make_branch_3.png)
    4. Make sure you are on your newly created branchMake sure you are on your. Otherwise 'checkout' your
       new branch. 
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/git_cola_make_branch_4.png)
4. (Only if you are done coding) Make a pull request.
    1. Add your code
    2. Commit you code
    3. Push upstream
 