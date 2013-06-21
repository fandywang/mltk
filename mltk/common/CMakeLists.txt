
MESSAGE(STATUS GTEST_ROOT=$ENV{GTEST_ROOT})

INCLUDE_DIRECTORIES(${MLTK_SOURCE_DIR})
INCLUDE_DIRECTORIES($ENV{GTEST_ROOT}/include)
LINK_DIRECTORIES($ENV{GTEST_ROOT}/lib)

ADD_EXECUTABLE(common_test
  double_vector_test.cc feature_test.cc feature_vocabulary_test.cc
  vocabulary_test.cc instance_test.cc mem_instance_test.cc)
TARGET_LINK_LIBRARIES(common_test gtest gtest_main)
TARGET_LINK_LIBRARIES(common_test ${CMAKE_THREAD_LIBS_INIT})

ADD_TEST(NAME common_test COMMAND common_test)
