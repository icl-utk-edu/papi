Summary: Performance Application Programming Interface
Name: papi
Version: 3.7.2
Release: 4%{?dist}
License: BSD
Group: Development/System
URL: http://icl.cs.utk.edu/papi/
Source0: http://icl.cs.utk.edu/projects/papi/downloads/%{name}-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root
BuildRequires: ncurses-devel
BuildRequires: gcc-gfortran
BuildRequires: kernel-devel >= 2.6.31
BuildRequires: chrpath
#Right now libpfm does not know anything about s390 and will fail
ExcludeArch: s390 s390x

%description
PAPI provides a programmer interface to monitor the performance of
running programs.

%package devel
Summary: Header files for the compiling programs with PAPI
Group: Development/System
Requires: papi = %{version}-%{release}
%description devel
PAPI-devel includes the C header files that specify the PAPI userspace
libraries and interfaces. This is required for rebuilding any program
that uses PAPI.

%prep
%setup -q

%build
cd src
PERF_HEAD=`ls /usr/src/kernels/*/include/linux/perf_counter.h|sort |tail -n 1` || exit 1
PERF_HEADER=`dirname $PERF_HEAD`
%configure --with-pcl=yes --with-pcl-incdir=$PERF_HEADER --disable-static
make %{?_smp_mflags}

#%check
#cd src
#make fulltest

%install
rm -rf $RPM_BUILD_ROOT
cd src
make DESTDIR=$RPM_BUILD_ROOT install

chrpath --delete $RPM_BUILD_ROOT%{_libdir}/*.so*

# Remove the static libraries. Static libraries are undesirable:
# https://fedoraproject.org/wiki/Packaging/Guidelines#Packaging_Static_Libraries
rm -rf $RPM_BUILD_ROOT%{_libdir}/*.a

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig
%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,-)
%{_bindir}/*
%{_libdir}/*.so.*
/usr/share/papi
%doc INSTALL.txt README LICENSE.txt RELEASENOTES.txt

%files devel
%defattr(-,root,root,-)
%{_includedir}/*.h
%{_includedir}/perfmon
%{_libdir}/*.so
%doc %{_mandir}/man3/*

%changelog
* Thu Nov 19 2009 William Cohen <wcohen@redhat.com> - 3.7.1-4
- Exclude s390 and s390x.

* Mon Nov 16 2009 William Cohen <wcohen@redhat.com> - 3.7.1-3
- Adjust configure.

* Mon Nov 16 2009 William Cohen <wcohen@redhat.com> - 3.7.1-2
- Bump version.

* Mon Nov 16 2009 William Cohen <wcohen@redhat.com> - 3.7.1-1
- Import papi-3.7.1.

* Mon Oct 26 2009 William Cohen <wcohen@redhat.com> - 3.7.0-11
- Add distro to release.

* Fri Oct 09 2009 William Cohen <wcohen@redhat.com> - 3.7.0-10
- Eliminate the papi-static subpackage.

* Fri Oct 09 2009 Richard W.M. Jones <rjones@redhat.com> - 3.7.0-9
- Fix defattr definitions.

* Fri Oct 09 2009 Richard W.M. Jones <rjones@redhat.com> - 3.7.0-8
- Fix URL and Source0.
- Grammatical corrections to the description sections.
- Remove RPATHs from shared libraries.
- RPM shouldn't own directories.
- Add soname patch so soname is libpapi.so.3.
- Add exit patch so library doesn't call exit directly.

* Thu Oct 01 2009 William Cohen <wcohen@redhat.com> - 3.7.0-7
- URL point to place to get download of release.
- Requires for -devel RPM.

* Tue Sep 29 2009 William Cohen <wcohen@redhat.com> - 3.7.0-6
- Remove the check section from the spec file.

* Tue Sep 29 2009 William Cohen <wcohen@redhat.com> - 3.7.0-5
- Compile x86_cache_info.c only on x86 machines.

* Fri Sep 25 2009 William Cohen <wcohen@redhat.com> - 3.7.0-4
- Add patch for multiplex.c.

* Thu Sep 24 2009 William Cohen <wcohen@redhat.com> - 3.7.0-3
- Add testing for C and Fortran.

* Thu Sep 24 2009 William Cohen <wcohen@redhat.com> - 3.7.0-2
- Split out the static libraries into separate sub package.

* Wed Sep 23 2009 William Cohen <wcohen@redhat.com> - 3.7.0-1
- Initial build.
