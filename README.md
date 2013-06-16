MLTK
==========================
The Machine Learning Toolkit (MLTK) is an open source C++ library supporting research and development in Machine Learning.

Depends
----------------------
1. [gflags](https://code.google.com/p/gflags/) - Commandline flags module for C++
Install to ~/thirdparty/gflags  
    ./configure --prefix=~/thirdparty/gflags  
    make & make install  
    export GFLAGS_ROOT="~/thirdparty/gflags"  
2. [glog](https://code.google.com/p/google-glog/) - Logging library for C++  
Install to ~/thirdparty/glog  
    ./configure --prefix=~/thirdparty/glog --with-gflags=~/thirdparty/gflags  
    make & make install  
    export GLOG_ROOT="~/thirdparty/glog"  
3. [gtest](https://code.google.com/p/googletest/) - Google C++ Testing Framework  
Install to ~/thirdparty/gtest  
    ./conifgure --prefix=~/thirdparty/gtest  
    make & make install  
    export GTEST_ROOT="~/thirdparty/gtest"  

Copyright and license
---------------------
Copyright (C) 2013 MLTK Project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this work except in compliance with the License.
You may obtain a copy of the License in the LICENSE file, or at:

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
