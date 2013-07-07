
SET(LIBRARY_OUTPUT_PATH ${MLTK_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${MLTK_SOURCE_DIR}/bin/mltk/common)

SET(SRC_LIST model_data.cc)

ADD_LIBRARY(mltk_common SHARED ${SRC_LIST})
SET_TARGET_PROPERTIES(mltk_common PROPERTIES CLEAN_DIRECT_OUTPUT 1)

ADD_LIBRARY(mltk_common_static STATIC ${SRC_LIST})
SET_TARGET_PROPERTIES(mltk_common_static PROPERTIES OUTPUT_NAME "mltk_common")
SET_TARGET_PROPERTIES(mltk_common_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

IF (test)
    INCLUDE_DIRECTORIES($ENV{GTEST_ROOT}/include)
    LINK_DIRECTORIES($ENV{GTEST_ROOT}/lib)

    ADD_EXECUTABLE(common_test
      double_vector_test.cc feature_test.cc feature_vocabulary_test.cc
      vocabulary_test.cc instance_test.cc mem_instance_test.cc model_data_test.cc)
    TARGET_LINK_LIBRARIES(common_test model_data gtest gtest_main)
    TARGET_LINK_LIBRARIES(common_test ${CMAKE_THREAD_LIBS_INIT})

    ADD_TEST(NAME common_test COMMAND ${EXECUTABLE_OUTPUT_PATH}/common_test)

    FILE(COPY testdata DESTINATION ${EXECUTABLE_OUTPUT_PATH})
ENDIF()
