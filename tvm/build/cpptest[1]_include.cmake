if(EXISTS "/home/nineis/ws/gpt-frontend/tvm/build/cpptest[1]_tests.cmake")
  include("/home/nineis/ws/gpt-frontend/tvm/build/cpptest[1]_tests.cmake")
else()
  add_test(cpptest_NOT_BUILT cpptest_NOT_BUILT)
endif()
