[#] start of __file__

AC_ARG_WITH([rocprofiler-ver],
            [AS_HELP_STRING([--with-rocprofiler-ver@<:@=ARG@:>@],
                            [Rocprofiler version to use in ROCm component (Default is 1)])],
            [with_rocprofiler_ver=$withval],
            [with_rocprofiler_ver=1])

if test "$with_rocprofiler_ver" = "1" ; then
    WITH_ROCPROFILER_VER=V1
else
    WITH_ROCPROFILER_VER=V2
fi

AC_SUBST([WITH_ROCPROFILER_VER])
AC_CONFIG_FILES([components/rocm/roc_profiler_config.h components/rocm/Rules.rocm])
