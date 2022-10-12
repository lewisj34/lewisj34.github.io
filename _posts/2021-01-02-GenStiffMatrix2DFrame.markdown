---
layout: post
title:  "Generalized Stiffness Matrix for 2D Frame Structures in C++"
date:   2021-01-02 12:12:52 -0600
categories: structures cpp
---
*<font size="5"> Basic Introduction</font>*

This is a short and sweet C++ framework to generate a global stiffness matrix for any 2D frame structure. Linear algebra library, [Armadillo][arma-docs], was used for matrix generation. Basic user input looks like this in `int main()`

{% highlight cpp %}
// instantiate frame object, f 
frame f; 

// element 1 
f.add_element(new element(
    2e8,        // E
    0.00048,    // I 
    0.075,      // A
    5,          // L
    7880,       // rho
    0,          // x1
    0,          // y1
    0,          // x2
    5,          // y2
    90));       // theta

// element 2 
f.add_element(new element(
    2e8,        // E
    0.00048,    // I 
    0.075,      // A
    5,          // L
    7880,       // rho
    0,          // x1
    5,          // y1
    0,          // x2
    10,         // y2
    90));       // theta

// create global stiffness matrix and print result
f.global_k();
{% endhighlight %}

Add elements to your frame structure using the `add_element` method of the `frame` class. Create the stiffness matrix and print the results using the `global_k` method of the `frame` class!

Nodes for the stiffness matrix are only representative of each element's end points, `x1`, `y1`, `x2`, and `x2`. 

*<font size="5"> Installation Instructions</font>*
1. Install linear algebra library, Armadillo [here][arma-download].
2. `git clone https://github.com/lewisj34/2DFrameStiffness.git` in the directory or location of your choice.
3. `cd buld && make && ./stiff` to run the example loaded. Expected output: 

{% highlight cpp %}
The global stifness matrix is shown below:

   9.2160e+03            0  -2.3040e+04  -9.2160e+03            0  -2.3040e+04            0            0            0            0            0            0
            0   3.0000e+06            0            0  -3.0000e+06            0            0            0            0            0            0            0
  -2.3040e+04            0   7.6800e+04   2.3040e+04            0   3.8400e+04            0            0            0            0            0            0
  -9.2160e+03            0   2.3040e+04   2.5184e+06            0            0  -9.2160e+03            0  -2.3040e+04  -2.5000e+06            0            0
            0  -3.0000e+06            0            0   6.0053e+06   1.6000e+04            0  -3.0000e+06            0            0  -5.3333e+03   1.6000e+04
  -2.3040e+04            0   3.8400e+04            0   1.6000e+04   2.1760e+05   2.3040e+04            0   3.8400e+04            0  -1.6000e+04   3.2000e+04
            0            0            0  -9.2160e+03            0   2.3040e+04   9.2160e+03            0   2.3040e+04            0            0            0
            0            0            0            0  -3.0000e+06            0            0   3.0000e+06            0            0            0            0
            0            0            0  -2.3040e+04            0   3.8400e+04   2.3040e+04            0   7.6800e+04            0            0            0
            0            0            0  -2.5000e+06            0            0            0            0            0   2.5000e+06            0            0
            0            0            0            0  -5.3333e+03  -1.6000e+04            0            0            0            0   5.3333e+03  -1.6000e+04
            0            0            0            0   1.6000e+04   3.2000e+04            0            0            0            0  -1.6000e+04   6.4000e+04
{% endhighlight %}

[arma-docs]: http://arma.sourceforge.net/docs.html#set_size
[arma-download]: http://arma.sourceforge.net/download.html
[src]: https://github.com/lewisj34/2DFrameStiffness