
SET(LIBRARY_OUTPUT_PATH ${MLTK_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${MLTK_SOURCE_DIR}/bin/common/base/string)

SET(SRC_LIST algorithm.cpp compare.cpp string_piece.cpp)

ADD_LIBRARY(base_string SHARED ${SRC_LIST})
SET_TARGET_PROPERTIES(base_string PROPERTIES CLEAN_DIRECT_OUTPUT 1)

ADD_LIBRARY(base_string_static STATIC ${SRC_LIST})
SET_TARGET_PROPERTIES(base_string_static PROPERTIES OUTPUT_NAME "base_string")
SET_TARGET_PROPERTIES(base_string_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)


INCLUDE_DIRECTORIES($ENV{GTEST_ROOT}/include)
LINK_DIRECTORIES($ENV{GTEST_ROOT}/lib)

ADD_EXECUTABLE(string_test algorithm_test.cpp string_piece_test.cpp)
TARGET_LINK_LIBRARIES(string_test base_string gtest gtest_main)
TARGET_LINK_LIBRARIES(string_test ${CMAKE_THREAD_LIBS_INIT})

ADD_TEST(NAME string_test COMMAND ${EXECUTABLE_OUTPUT_PATH}/string_test)

