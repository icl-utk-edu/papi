struct wininfo {
	unsigned int family;
	unsigned int brand_id;
	unsigned int model;
	unsigned int processor_type;
	unsigned int ncpu;
	unsigned int nnodes;
	unsigned int mhz;
	unsigned int revision;
	unsigned short int arch, total_cpus;
	unsigned int L2cache_size;
	unsigned int myvendor;
	unsigned int nrctr;
	int vendor;
	char vendor_string[16];
	char model_string[40];
};

#define UNKNOWN 0
#define INTEL   1
#define AMD     2

// Macros for processor types
#define IS_UNKNOWN(hwinfo)		((hwinfo)->myvendor==UNKNOWN)
#define IS_AMD(hwinfo)			((hwinfo)->myvendor==AMD)
#define IS_INTEL(hwinfo)		((hwinfo)->myvendor==INTEL)

// AMD processors0
#define IS_AMDATHLON(hwinfo)   (IS_AMD((hwinfo))&&(hwinfo)->family==6&&((hwinfo)->model==1\
								  ||(hwinfo)->model==2||(hwinfo)->model==4))
#define IS_AMDDURON(hwinfo )   (IS_AMD((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==3)

// Intel Processors
#define IS_P4(hwinfo)			(IS_INTEL((hwinfo))&&(hwinfo)->family==15&&(hwinfo)->model==0)
#define IS_P3XEON(hwinfo)		(IS_P3XEONA((hwinfo))||IS_P3XEON8((hwinfo))||IS_P3XEON7((hwinfo)))
#define IS_P3XEONA(hwinfo)		(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==10)
#define IS_P3XEON8(hwinfo)		(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==8\
								&&(hwinfo)->brand_id==3)
#define IS_P3XEON7(hwinfo)      (IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==7\
								&&((hwinfo)->L2cache_size>=1024))
#define IS_P3(hwinfo)			(IS_P38((hwinfo))||IS_P37((hwinfo)))
#define IS_P38(hwinfo)			(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==8\
								&&(hwinfo)->brand_id==2)
#define IS_P37(hwinfo)          (IS_INTEL(hwinfo)&&(hwinfo)->family==6&&(hwinfo)->model==7\
								&&((hwinfo)->L2cache_size>0 && (hwinfo)->L2cache_size <=512))
#define IS_CELERON(hwinfo)		(IS_CELERON8((hwinfo))||IS_CELERON6((hwinfo))||IS_CELERON5((hwinfo)))
#define IS_CELERON8(hwinfo)		(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==8\
								&&(hwinfo)->brand_id==1)
#define IS_CELERON5(hwinfo)     (IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==5\
								&&(hwinfo)->L2cache_size == 0 )
#define IS_CELERON6(hwinfo)		(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==6\
								&&(hwinfo)->L2cache_size<256)
#define IS_P2(hwinfo)			(IS_P25((hwinfo))||IS_P23((hwinfo))||IS_P26((hwinfo)))
#define IS_P23(hwinfo)			(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==3)
#define IS_P25(hwinfo)          (IS_INTEL(hwinfo)&&(hwinfo)->family==6&&(hwinfo)->model==5\
								&&(hwinfo)->L2cache_size>0&&(hwinfo)->L2cache_size<1024)
#define IS_P26(hwinfo)			(IS_INTEL((hwinfo))&&(hwinfo)->family==6&&(hwinfo)->model==6\
								&&(hwinfo)->L2cache_size>=256)
#define IS_P2XEON(hwinfo)       (IS_INTEL(hwinfo)&&(hwinfo)->family==6&&(hwinfo)->model==5\
								&&(hwinfo)->L2cache_size>=1024)

int init_hwinfo( struct wininfo * hwinfo);