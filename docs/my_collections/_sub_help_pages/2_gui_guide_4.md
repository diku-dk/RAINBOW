---
layout: sub_page
title: "GUI guide"
subtitle: "Make Unittest"
role: "child"
prefix: "contribute"
postfix: "gui_guide/make_unittest"
permalink: my_collections/sub_help_pages/contribute/gui_guide/make_unittest
---
# Make Unittest
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