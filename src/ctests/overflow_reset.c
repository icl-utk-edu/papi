/* Contributed by harald@cepba.upc.edu, modified by Philip Mucci */
#include "papi_test.h"

void handler (int EventSet, void *address, long_long overflow_vector, void *context)
{
	unsigned long long vals[8];
	int ret;

  printf ("Overflow at %p! bit=0x%llx \n", address, overflow_vector);
	ret = PAPI_read (EventSet, vals);
}

int main (int argc, char *argv[])
{
	int EventSet = PAPI_NULL;
	int retval, i;
	PAPI_option_t options;
	PAPI_option_t options2;

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT && retval > 0)
	{ printf("PAPI library version mismatch!\n"); return 1; }

	retval = PAPI_get_opt (PAPI_HWINFO, &options);
	if (retval < 0)
	{printf ("PAPI_get_opt failed\n"); return 0; }
	printf ("ovf_info = %d (%x)\n", options.ovf_info.type, options.ovf_info.type);

	retval = PAPI_get_opt (PAPI_SUBSTRATEINFO/*PAPI_SUBSTRATE_SUPPORT*/, &options2);
	if (retval < 0)
	{printf ("PAPI_get_opt failed\n"); return 0; }
/*	printf ("sub_info.supports_hw_overflow = %d\n", options2.sub_info.supports_hw_overflow); */
	printf ("sub_info.hardware_intr = %d\n", options2.sub_info->hardware_intr);

	retval = PAPI_create_eventset (&EventSet);
	if (retval < 0)
	{printf ("PAPI_create_eventset failed\n"); return 0; }

	retval = PAPI_add_event (EventSet, PAPI_TOT_INS);
	if (retval < 0)
	{printf ("PAPI_add_event failed\n"); return 0; }
	retval = PAPI_add_event (EventSet, PAPI_TOT_CYC);
	if (retval < 0)
	{printf ("PAPI_add_event failed\n"); return 0; }

	retval = PAPI_overflow (EventSet, PAPI_TOT_INS, 10000000, 0, handler);
	if (retval < 0)
	{printf ("PAPI_overflow failed\n"); return 0; }

	PAPI_start (EventSet);

	for (i = 0; i < 1000000; i++)
	{
		if (i%1000 == 0)
		{
			long_long merda[8];
			int i;
			long_long vals[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

			PAPI_read (EventSet, vals);
#if 1
			PAPI_reset (EventSet);
#endif
			
			printf ("Main loop read vals :");
			for (i = 0; i < 3 /* 8 */; i++)
				printf ("%lld ", vals[i]);	
			printf ("\n");
		}
	}

	return 0;
}
