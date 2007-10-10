Name: perfctr
Summary: Linux performance monitoring counters software
Version: 2.7.21.1
Release: 1
License: LGPL
Group: Development/Tools
URL: http://user.it.uu.se/~mikpe/linux/perfctr/
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot
Source: %{name}-%{version}.tar.gz

%description
This package adds support for using the Performance-Monitoring
Counters (PMCs) found in many modern processors.

PMCs are "event counters" capable of recording any of a large
number of performance-related events during execution.
These events typically include instructions executed, cache
misses, TLB misses, stalls, and other events specific to
the microarchitecture of the processor being used.

PMCs are primarily used to identify low-level performance problems,
and to validate code changes intended to improve performance.

%package devel
Summary: Development headers and libraries for perfctr
Group: Development/Libraries

%description devel
The perfctr-devel package contains the header and object files
necessary for developing programs which use the perfctr C library.

%prep
%setup -q

%build
make

%install
rm -rf %{buildroot}
make install2 \
	PREFIX=%{buildroot}/%{_prefix} \
	BINDIR=%{buildroot}/%{_bindir} \
	LIBDIR=%{buildroot}/%{_libdir} \
	INCLDIR=%{buildroot}/%{_includedir}
/sbin/ldconfig -n %{buildroot}/%{_libdir}

%clean
rm -rf %{buildroot}

%files
%defattr(-,root,root,-)
%{_bindir}/perfex
%{_libdir}/*.so*

%doc README CHANGES TODO OTHER

%post
/sbin/ldconfig

%postun -p /sbin/ldconfig

%files devel
%defattr(-,root,root,-)
%{_includedir}/*.h
%{_includedir}/*/*.h
%{_libdir}/*.a


%changelog
* Wed Oct 03 2007 Mikael Pettersson <mikpe@it.uu.se> -
- Corrected email address and URL.

* Fri Apr 08 2005 Mikael Pettersson <mikpe@csd.uu.se> -
- Don't create /dev/perfctr or update /etc/modules.conf.

* Tue Sep 16 2004 Mikael Pettersson <mikpe@csd.uu.se> -
- Dropped obsolete x86 qualification from Summary.

* Sun Dec 21 2003 Mikael Pettersson <mikpe@csd.uu.se> -
- Create /dev/perfctr in %post, not in %install and %files.
  This avoids incorrect deletion of the node on package uninstall.
- Don't add alias to /etc/modules.conf if it's already there.

* Sun Nov 23 2003 Mikael Pettersson <mikpe@csd.uu.se> -
- libperfctr.so install and uninstall fixes.

* Tue Sep 16 2003 Mikael Pettersson <mikpe@csd.uu.se> -
- No longer necessary to add module alias to /etc/modprobe.conf.

* Wed Jul 03 2003 Bryan O'Sullivan <bos@serpentine.com> -
- Fix module files for both 2.4 and 2.5 kernels.

* Wed Jul 02 2003 Mikael Pettersson <mikpe@csd.uu.se> -
- Corrected License and URL fields.

* Mon Jun 16 2003 Bryan O'Sullivan <bos@serpentine.com> -
- Add device file.
- Add module alias.

* Thu Jun 12 2003 Bryan O'Sullivan <bos@serpentine.com> - 
- Initial build.
