Summary: a performance monitoring library for Linux/ia64
Name: libpfm
Version: 3.0
Release: 1
License: MIT-style
Group: Development/Libraries
ExclusiveArch: ia64
ExclusiveOS: linux
AutoReqProv: no
Packager: <eranian@hpl.hp.com>
Vendor: Hewlett-Packard Company
Source: ftp://ftp.hpl.hp.com/pub/linux-ia64/%{name}-%{version}.tar.gz
Prefix: %{_prefix}
BuildRoot: %{_tmppath}/%{name}-%{version}-root



%description
This package contains a library to develop performance monitoring 
applications using the IA-64 Performance Monitor Unit (PMU). 
This version supports both the Itanium and Itanium 2 processors.

%package devel
Summary: the Linux/ia64 performance library (libpfm) development files.
Group: Development/Libraries

%description devel
The performance monitoring library (libpfm) is used to develop 
performance monitoring applications using the IA-64 Performance Monitoring 
Unit (PMU). This package provides the files necessary for development of
applications. This version supports both the Itanium and Itanium 2 processors.
It requires a kernel perfmon-2.x subsystem.

%prep
%setup -q
%install
rm -rf %{buildroot}
mkdir -p %{buildroot}/%{_prefix}
make install DESTDIR=%{buildroot}/%{_prefix}

%post
/sbin/ldconfig
%preun
%postun
/sbin/ldconfig
%clean
rm -rf %{buildroot}

%files
%doc README
%defattr(-,root,root)
%attr(755,root,root) %{_prefix}/lib/libpfm.so.%{PACKAGE_VERSION}.0


%files devel
%doc README
%doc examples/multiplex.c
%doc examples/ita2_irr.c
%doc examples/ita2_opcode.c
%doc examples/ita2_rr.c
%doc examples/ita_btb.c
%doc examples/ita_irr.c
%doc examples/ita_opcode.c
%doc examples/ita_rr.c
%doc examples/notify_self.c
%doc examples/notify_self2.c
%doc examples/notify_self3.c
%doc examples/self.c
%doc examples/showreset.c
%doc examples/syst.c
%doc examples/task.c
%doc examples/whichpmu.c
%doc examples/ita2_btb.c
%doc examples/ita2_dear.c
%doc examples/ita_dear.c
%doc examples/task_attach.c
%doc examples/task_attach_timeout.c
%doc examples/task_smpl.c
%attr(644,root,root) %{_prefix}/lib/libpfm.a
%attr(644,root,root) %{_prefix}/include/perfmon/perfmon.h
%attr(644,root,root) %{_prefix}/include/perfmon/perfmon_default_smpl.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_comp.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_comp_ia64.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_os.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_os_ia64.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_generic_ia64.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_itanium.h
%attr(644,root,root) %{_prefix}/include/perfmon/pfmlib_itanium2.h
%attr(644,root,root) %{_mandir}/man3/*

%changelog
* Mon Dec 1  2003 Stephane Eranian <eranian@hpl.hp.com>
- release 3.0 final version
* Thu Jan 02  2003 Stephane Eranian <eranian@hpl.hp.com>
- release final 2.0 version
* Fri Dec 20  2002 Stephane Eranian <eranian@hpl.hp.com>
- final 2.0 release
* Thu Dec 05  2002 Stephane Eranian <eranian@hpl.hp.com>
- initial release of the library as a standalone package
- see ChangeLog for actual log
