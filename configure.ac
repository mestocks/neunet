AC_PREREQ([2.69])
AC_INIT([neunet], [0.0.9], [mestocks@protonmail.com])

AC_PROG_CC
AC_PROG_INSTALL
AC_TYPE_UINT64_T
AC_FUNC_MALLOC

LT_INIT

AC_ENABLE_SHARED
AC_DISABLE_STATIC
AC_CONFIG_FILES([Makefile])
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_SRCDIR([src/main.c])

AC_CHECK_HEADERS([assert.h limits.h math.h stdint.h stdlib.h string.h time.h unistd.h sys/stat.h])
AC_CHECK_HEADERS([liblar/lar_algebra.h liblar/lar_init.h liblar/lar_objects.h], [], [], [])

AC_CHECK_LIB([m],[exp],,[])
AC_CHECK_LIB([lar],[lar_matrix_addition],,[])

AM_INIT_AUTOMAKE([subdir-objects foreign])

AC_OUTPUT




