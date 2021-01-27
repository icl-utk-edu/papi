prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=@CMAKE_INSTALL_PREFIX@/@RUNTIME_INSTALL_DIR@
libdir=@CMAKE_INSTALL_PREFIX@/@LIB_INSTALL_DIR@
includedir=@CMAKE_INSTALL_PREFIX@/@INCLUDE_INSTALL_DIR@

Name: PAPI
Description: Performance API to access performance metrics on system
Version: @PACKAGE_VERSION@
Libs: -L\${libdir} -lpapi
Libs.private: @LIBS@
Cflags: -I\${includedir}

