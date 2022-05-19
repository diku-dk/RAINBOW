---
layout: page
title: "How to contribute?"
prefix: "contribute"
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
`git checkout <issue_number>b`\
Where the issue number would be **, see GUI.
4. Make pull request: \
`git push --set-upstream origin <issue_number>b` 
5. Use the GUI guide
6. Use the GUI guide
7. Use the GUI guide

### Example GUI Git Desktop
#### Download & install
1. Go to the libRainbow github webpage: [Click here!](https://github.com/diku-dk/libRAINBOW)  
![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/1.png)
2. Find the "Open with GitHub Desktop" in the "code" dropdown.
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/2.png)
3. Press "Open URL:x-github-client" in the following popup
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/3.png)
4. Wait for the download to finish. Thereafter execute the installation 
   program
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/4.png)
5. Open the "GitHub desktop" app. Control that your working repository is the
   libRainbow
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/5.png)
    5.1 **In Case your working repository is not libRainbow** 
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/6.png)
    5.2 Press "Sign-in" 
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/7.png)
    5.2 Sign-in with your github account. **Recommendation**: Use firefox or chrome 
    for the sign-in browser
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/8.png)
9. Press "Authorize desktop"
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/9.png)
10. Press "Open GitHubDesktop.exe"
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/10.png)
11. Congratulation, you are done with the installation. If it does not work, don't 
hesitate to contact the maintainer
 ![]({{ site.url }}{{ site.baseurl }}/assets/images/installation/11.png)

12. Next: Before uploading your code, you need to create an issue and 
create a branch. The main branch is not for development. 

#### Create Issue and Branch
1. First open "GitHub Desktop". Then, 
Find the "Create issue on GitHub" button from the the "Repository"
dropdown
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/1.png)
2. Follow the link to the GitHub webpage. 
    1. Write title. Describe the problem you are trying to solve 
    in one sentence
    2. Write describtion. Elaborate on what your problem is. 
    State any crucial assumptions. 
    3. Assign Assignees 
    4. Assign label
    5. Submit issue 
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/2.png)
3. Remember the issue number. The issue number will be needed for the 
branch name. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/3.png)
4. Find the "New branch" button in the "Branch" dropdown
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/4.png)
5. Create branch page
    1. Name: Use the issue number: "< issue number >b""
    2. Make sure, that you branch from the main branch
    3. Press "Create branch"
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/5.png)
6. Press "Publish branch" 
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/6.png)
7. Select your branch
![]({{ site.url }}{{ site.baseurl }}/assets/images/issue/7.png)

#### Upload code
1. We have coded a new simulator that we want to submit to the libRAINBOW repository for this example.  
Therefore we start by navigating to the simulator folder. Then, if you have made another implementation, such as a mathematical module, you place it in the math folder.
In the simulation folder, create a new folder with your implementation name. In this case, "new_simulator".  
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial.png)
2. Then, in your newly created folder, insert your implementation. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_2.png)
3. Open "GitHub Desktop". Ensure that all your implementation is visible in "changed file". 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_3.png)
4. Write title use the format: < issue number >: your commit messages. Thereafter, write a short 
description. Then press "commit".
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_4.png)
5. Press "Push origin"
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_5.png)
6. Return to the GitHub webpage. Press "Compare & pull results" 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_6.png)
8. Define "pull results"
    1. Assign Reviewers
    2. Assign Assignees
    3. Assign Label
    4. Assign Issue
    5. Before merge -> All revierwers mus approve
    6. Before merge -> All tests must pass 
    7. Then press "Squash and merge" 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_8.png)
9. Next, adding unittest to your code.

#### Make Unittest
1. Consider the following module. We have created two functions for testing—a function that works and a function that fails. The "Correct" function "simulator_func" and the "Incorrect"
function "simulator_func_error".
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_1.png)
2. First, insert the module in the correct folder. In this case 
libRAINBOW > python > isl > simulators > new_simulator
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_2.png)
3. Then adding test. Navigate to the test folder.  
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_3.png)
4. In the test folder. Create a new folder, use the format: < test_< your name > >. In this 
case: test_new_simulator
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_4.png)
5. Navigate to your newly created folder
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_5.png)
6. Create a test file using the format < test_ < your name >>. In this case test_new_simulator.
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_6.png)
7. In your test file. Copy all the imports and sys.path..
 To test your implementation, you need to import it:
isl.simulators.< Your implementation folder >.< Simulator name> as sn.
In this case:
```
    isl.simulators.new_simulators.brandNewSimulator as bns
```
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_7.png)

8. We use the unittest [framework](https://docs.python.org/3/library/unittest.html), therefore
you need to define a test class. 
First, inherit from the "unittest" module. Then every member function must start with the prefix "test_". Third, give the self variable as parameter
to all member functions. For adding the test. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_9.png)
9. For testing a Numpy array, start by importing the Numpy library. Then import "utils". Finally, to test if two arrays are equal, use the "array_equal".
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_10.png)
10. Ensure that all your test and implementation is shown in "GitHub Desktop"
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_11.png)
11. Make a commit to you branch. Use the same name format as before
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_12.png)
12. Press "Push Origin"
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_13.png)
13. In case of a test fail, click details. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_14.png)
14. If all test passed and all reviewers approves, you are allowed to merge
![]({{ site.url }}{{ site.baseurl }}/assets/images/unittest/adding_test_tutorial_15.png)

#### Documentation
1. Change to the "gh-pages" branch 
![]({{ site.url }}{{ site.baseurl }}/assets/images/documentation/1.png)
2. Navigate to "_documentation_pages" folder
![]({{ site.url }}{{ site.baseurl }}/assets/images/documentation/2.png)
3. Copy paste the .template folder. Rename the copy using the format:
```< folder name >_< file name >_< function name >.md```
![]({{ site.url }}{{ site.baseurl }}/assets/images/documentation/3.png)
4. The template should look like this
![]({{ site.url }}{{ site.baseurl }}/assets/images/documentation/4.png)
5. Type the needed information eg. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/documentation/5.png) 