
SET(LIBRARY_OUTPUT_PATH ${MLTK_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${MLTK_SOURCE_DIR}/bin/mltk/maxent)

SET(SRC_LIST maxent.cc lbfgs.cc owlqn.cc sgd.cc)

ADD_LIBRARY(maxent SHARED ${SRC_LIST})
SET_TARGET_PROPERTIES(maxent PROPERTIES CLEAN_DIRECT_OUTPUT 1)
TARGET_LINK_LIBRARIES(maxent model_data)

ADD_LIBRARY(maxent_static STATIC ${SRC_LIST})
SET_TARGET_PROPERTIES(maxent_static PROPERTIES OUTPUT_NAME "maxent")
SET_TARGET_PROPERTIES(maxent_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
TARGET_LINK_LIBRARIES(maxent_static model_data)

IF (test)
    INCLUDE_DIRECTORIES($ENV{GTEST_ROOT}/include)
    LINK_DIRECTORIES($ENV{GTEST_ROOT}/lib)

    ADD_EXECUTABLE(maxent_test maxent_test.cc)
    TARGET_LINK_LIBRARIES(maxent_test maxent gtest gtest_main)
    TARGET_LINK_LIBRARIES(maxent_test ${CMAKE_THREAD_LIBS_INIT})

    ADD_TEST(NAME maxent_test COMMAND ${EXECUTABLE_OUTPUT_PATH}/maxent_test)
ENDIF()

INCLUDE_DIRECTORIES($ENV{GFLAGS_ROOT}/include)
LINK_DIRECTORIES($ENV{GFLAGS_ROOT}/lib)
INCLUDE_DIRECTORIES($ENV{GLOG_ROOT}/include)
LINK_DIRECTORIES($ENV{GLOG_ROOT}/lib)

ADD_EXECUTABLE(maxent_trainer maxent_trainer_main.cc)
TARGET_LINK_LIBRARIES(maxent_trainer maxent base_string gflags glog)

ADD_EXECUTABLE(maxent_predictor maxent_predictor_main.cc)
TARGET_LINK_LIBRARIES(maxent_predictor maxent base_string gflags glog)
