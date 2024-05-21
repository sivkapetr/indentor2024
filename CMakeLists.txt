cmake_minimum_required(VERSION 3.14)

project(Examples DESCRIPTION "Usage AniFem++ example" LANGUAGES CXX)

find_package(anifem++ REQUIRED)

add_executable(hyperelast nonlin_elast.cpp prob_args.cpp prob_args.h example_common.cpp example_common.h)
target_link_libraries(hyperelast AniFem++::AniFem++)
