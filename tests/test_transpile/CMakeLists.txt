cmake_minimum_required(VERSION 3.0)

project("Loki C-transpile style-demonstrator")
enable_language(C Fortran)

# Enable needed modules and languages
enable_language(Fortran)

add_executable(simple_loop simple_loop.f90 simple_loop_c.c)

