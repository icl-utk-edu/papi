

#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "papi_test.h"

#define MAX_COMPONENTS 8

int EventSet[MAX_COMPONENTS];
int NumEvents[MAX_COMPONENTS];


int papi_xml_hwinfo(FILE *f)
{
  const PAPI_hw_info_t *hwinfo;
  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    return(PAPI_ESBSTR);
  
  fprintf(f, "<hardware>\n");
  fprintf(f, "  <vendor string=\"%s\" code=\"%d\"/>\n", 
	  hwinfo->vendor_string, hwinfo->vendor);
  fprintf(f, "  <model string=\"%s\" code=\"%d\"/>\n",
	  hwinfo->model_string, hwinfo->model);
  fprintf(f, "  <system nodes=\"%d\" totalCPUs=\"%d\"/>\n",
	  hwinfo->nnodes, hwinfo->totalcpus);
  fprintf(f, "  <node CPUs=\"%d\"/>\n",
	  hwinfo->ncpu);
  fprintf(f, "  <CPU revision=\"%f\" clockrate=\"%f\" />\n",
	  hwinfo->revision, hwinfo->mhz);
  fprintf(f, "  <clock rate=\"%d\" tickspersec=\"%d\" />\n",
	  hwinfo->clock_mhz, hwinfo->clock_ticks );

  fprintf(f, "</hardware>\n");

  return (PAPI_OK);
}


void papi_init(int argc, char **argv)
{
  int i;
  int retval;
  const PAPI_hw_info_t *hwinfo = NULL;
  
  tests_quiet(argc, argv);     /* Set TESTS_QUIET variable */
  
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    test_fail(__FILE__, __LINE__, "PAPI_library_init", retval);
  
  retval = PAPI_set_debug(PAPI_VERB_ECONT);
  if (retval != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_set_debug", retval);
  
  
  for( i=0; i<MAX_COMPONENTS; i++ )
    {
      EventSet[i]=PAPI_NULL;
      NumEvents[i]=0;
      retval=PAPI_create_eventset(&(EventSet[i]));
      if(retval != PAPI_OK){
	fprintf(stderr, "PAPI_create_eventset error\n");
	exit(1);
      }
    }

}

/* 
   1  : can be added 
   0  : cannot be added 
*/
int test_event( int cidx, int evt )
{
  int retval; 

  retval = PAPI_add_event(EventSet[cidx], evt);
  if ( retval != PAPI_OK ) {
    // fprintf(stdout, "*** Cannot add %d\n", evt);
    return 0;
  }
  
  if((retval=PAPI_remove_event(EventSet[cidx], evt))!=PAPI_OK)
    {
      fprintf(stderr, "Error removing event from eventset\n");
      exit(1);
    }
  return 1;
}

char* component_type(char *id)
{
  char *str=strdup(id);
  char *s=str+5;
  
  while( *s && *s!='.' )
    s++;
  
  *s=0;
  return str+5;
}


void enum_events(FILE *f, int cidx, int modifier)
{
  int i, k;
  int retval;
  PAPI_event_info_t info;
  const PAPI_component_info_t *comp;

  comp = PAPI_get_component_info(cidx);
  i = PAPI_COMPONENT_MASK(cidx)|modifier;

  fprintf(f, "<component index=\"%d\" type=\"%s\" id=\"%s\">\n", 
	  cidx, cidx?component_type(comp->name):"CPU", comp->name );
  fprintf(f, "  <eventset type=\"%s\">\n", 
	  modifier&PAPI_PRESET_MASK?"PRESET":"NATIVE" );
  
  retval=PAPI_enum_event(&i, PAPI_ENUM_FIRST);
  while( retval==PAPI_OK )
    {
      retval = PAPI_get_event_info(i, &info);
      if( retval!=PAPI_OK ) {
	retval = PAPI_enum_event(&i, PAPI_ENUM_EVENTS);
	continue;
      }

      if( !test_event(cidx, i) )
	{
	  retval = PAPI_enum_event(&i, PAPI_ENUM_EVENTS);
	  continue;
	}

      fprintf(f, "    <event name=\"%s\" desc=\"%s\" code=\"0x%x\">\n",
	      info.symbol, info.long_descr, info.event_code);

      

      if( modifier&PAPI_NATIVE_MASK )
	{
	  if (comp->cntr_umasks) {
	    k = i;
	    if (PAPI_enum_event(&k, PAPI_NTV_ENUM_UMASKS) == PAPI_OK) {
	      do {
		retval = PAPI_get_event_info(k, &info);
		if (retval == PAPI_OK) {
		  		  
		  if( !test_event(cidx, k) )
		    {
		      retval = PAPI_enum_event(&i, PAPI_ENUM_EVENTS);
		      continue;
		    }

		  fprintf(f, "        <modifier name=\"%s\" desc=\"%s\" code=\"0x%x\"> </modifier>\n",
			  strchr(info.symbol, ':'),
			  strchr(info.long_descr, ':')+1, 
			  info.event_code );

		  
		}
	      } while (PAPI_enum_event(&k, PAPI_NTV_ENUM_UMASKS) == PAPI_OK);
	    }
	  }
	}

      fprintf(f, "    </event>\n" );

      retval = PAPI_enum_event(&i, PAPI_ENUM_EVENTS);
    }

  fprintf(f, "  </eventset>\n");
  fprintf(f, "</component>\n");

  
}



void usage( int argc, char *argv[] )
{
  fprintf(stderr, "Usage: %s [options] [[event1] event2 ...]\n", argv[0]);
  fprintf(stderr, "     options: -h     print help message\n");
  fprintf(stderr, "              -p     print only preset events\n");
  fprintf(stderr, "              -n     print only native events\n");
  fprintf(stderr, "              -c n   print only events for component index n\n");
}


int main( int argc, char *argv[] )
{
  int i;
  int retval;

  int comp=-1;
  int numc=0;

  int preset=-1;
  int native=-1;

  papi_init( argc, argv );

  numc = PAPI_num_components();  

  for( i=1; i<argc; i++ )
    {
      if( argv[i][0]=='-' )
	{
	  switch( argv[i][1] )
	    {
	    case 'c':
	      comp=(i+1)<argc?atoi(argv[(i++)+1]):-1;
	      if( comp<0 || comp>=numc )
		{
		  fprintf(stderr, "Error: component index %d out of bounds (0..%d)\n",
			  comp, numc-1);
		  usage(argc, argv);
		  return 1;
		}
	      break;

	    case 'p':
	      preset=1;
	      native=(native>0?1:0);
	      break;

	    case 'n':
	      native=1;
	      preset=(preset>0?1:0);
	      break;
	      
	    case 'h':
	      usage(argc, argv);
	      return 0;
	      break;

	    default:
	      fprintf(stderr, "Error: unknown option: %s\n", argv[i]);
	      usage(argc, argv);
	      return 1;
	    }
	}
      else
	{
	  int code=-1; 
	  int cidx=0;
	  
	  retval = PAPI_event_name_to_code(argv[i], &code);
	  retval = PAPI_query_event( code );
	  if( retval!=PAPI_OK )
	    {
	      fprintf(stderr, "Error: unknown event: %s\n", argv[i]);
	      usage(argc, argv);
	      return 1;
	    }

	  cidx = PAPI_COMPONENT_INDEX(code);
	  retval = PAPI_add_event(EventSet[cidx], code);
	  if ( retval != PAPI_OK )
	    {
	      fprintf(stderr, "Error: event %s cannot be counted with others\n", argv[i]);
	      usage(argc, argv);
	      return 1;
	    }
	  else
	    NumEvents[cidx]++;
	}
    }

  if( native<0 && preset<0 )
    native=preset=1;
  
  fprintf(stdout, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  fprintf(stdout, "<eventinfo>\n");
  
  papi_xml_hwinfo(stdout);
  
  if( comp>=0 )
    {
      if( native )
	enum_events(stdout, comp, PAPI_NATIVE_MASK);
      if( preset )
	enum_events(stdout, comp, PAPI_PRESET_MASK);
    }
  else
    {
      for( i=0; i<numc; i++ )
	{
	  if( native )
	    enum_events(stdout, i, PAPI_NATIVE_MASK);
	  if( preset )
	    enum_events(stdout, i, PAPI_PRESET_MASK);

	  //fprintf(stderr, "nc=%d %s\n", i, info->name);
	  //fprintf(stderr, "nc=%d %s\n", i, info->version);
	  //fprintf(stderr, "nc=%d %d\n", i, info->num_preset_events );
	  //fprintf(stderr, "nc=%d %d\n", i, info->num_native_events );
	}
    }
  fprintf(stdout, "</eventinfo>\n");

  return 0;
}
