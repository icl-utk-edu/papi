Summary: Performance Application Programming Interface
Name: papi
Version: 4.0.0
Release: 1%{?dist}
License: BSD
Group: Development/System
URL: http://icl.cs.utk.edu/papi/
Source0: http://icl.cs.utk.edu/projects/papi/downloads/%{name}-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root
BuildRequires: ncurses-devel
BuildRequires: gcc-gfortran
BuildRequires: kernel-headers >= 2.6.31
BuildRequires: chrpath
#Right now libpfm does not know anything about s390 and will fail
ExcludeArch: s390, s390x

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
%configure --disable-static
make

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
%doc %{_mandir}/man1/*

%changelog
* Wed Jan 13 2010 William Cohen <wcohen@redhat.com> - 4.0.0-1
- Generate papi.spec file for papi-4.0.0.
