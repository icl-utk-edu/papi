AC_INIT(topdown, version-0.1)

AC_PROG_CC

AC_ARG_WITH([librseq], 
			AS_HELP_STRING([--without-librseq], [Ignore the presence of librseq and disable rseq protections for heterogeneous processors]))

AC_ARG_WITH([librseq-install-path], 
			[AS_HELP_STRING([--with-librseq-install-path], 
				[install location of librseq, defaults to /usr/local/lib])],
			[RSEQ_LDFLAGS="-L$withval"],
			[RSEQ_LDFLAGS="-L/usr/local/lib"])
AC_SUBST([RSEQ_LDFLAGS])

AS_IF([test "x$with_librseq" != "xno"],
	[AC_CHECK_LIB(rseq, rseq_available, [have_librseq=yes])],
	[have_librseq=no; RSEQ_LDFLAGS=""])

AS_IF([test "x$have_librseq" = "xyes"],
	[RSEQ_LDLIBS="-lrseq"; RSEQ_DEFINES="-DLIBRSEQ"],
	[AS_IF([test "x$with_librseq" = "xyes"],
		[AC_MSG_ERROR([librseq requested but not found])
	])
])
AC_SUBST([RSEQ_LDLIBS])
AC_SUBST([RSEQ_DEFINES])


AC_CONFIG_FILES([Makefile.topdown])
AC_OUTPUT
