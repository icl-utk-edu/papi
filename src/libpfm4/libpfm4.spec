%define python_sitearch %(python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(1)")
%define python_prefix %(python -c "import sys; print sys.prefix")

Name:		libpfm
Version:	4.2.0
Release:	0%{?dist}

Summary:	Library to encode performance events for use by perf tool

Group:		System Environment/Libraries
License:	MIT
URL:		http://perfmon2.sourceforge.net/
Source0:	http://sourceforge.net/projects/perfmon2/files/libpfm4/%{name}-%{version}.tar.gz
BuildRequires:	python-devel
BuildRequires:	python-setuptools-devel
BuildRequires:  swig
BuildRoot:	%{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

%description

libpfm4 is a library to help encode events for use with operating system
kernels performance monitoring interfaces. The current version provides support
for the perf_events interface available in upstream Linux kernels since v2.6.31.

%package devel
Summary:	Development library to encode performance events for perf_events based tools
Group:		Development/Libraries
Requires:	%{name} = %{version}

%description devel
Development library and header files to create performance monitoring applications for the
perf_events interface.

%package python
Summary:	Python bindings for libpfm4 and perf_event_open system call.
Group:		Development/Languages
Requires:	%{name} = %{version}

%description python
Python bindings for libpfm4 and perf_event_open system call.

%prep
%setup -q

%build
make

%install
rm -rf $RPM_BUILD_ROOT
make \
    PREFIX=$RPM_BUILD_ROOT%{_prefix} \
    LIBDIR=$RPM_BUILD_ROOT%{_libdir} \
    PYTHON_PREFIX=$RPM_BUILD_ROOT/%{python_prefix} \
    install

%clean
rm -fr $RPM_BUILD_ROOT

%post -p /sbin/ldconfig
%postun	-p /sbin/ldconfig

%files
%defattr(644,root,root,755)
%doc README
%attr(755,root,root) %{_libdir}/lib*.so*

%files devel
%defattr(644,root,root,755)
%{_includedir}/*
%{_mandir}/*
%{_libdir}/lib*.a

%files python
%defattr(644,root,root,755)
%attr(755,root,root) %{python_sitearch}/*

%changelog
* Wed Jan 12 2011 Arun Sharma <asharma@fb.com> 1.0
Initial revision
