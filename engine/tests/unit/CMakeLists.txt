add_executable(test_costs  test_costs.cpp)
add_executable(test_gateway test_gateway.cpp)

foreach(t test_costs test_gateway)
    target_link_libraries(${t} PRIVATE juniauto_core)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
