

#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "papi_test.h"

#if PAPI_VERSION_MAJOR(PAPI_VERSION)>=3 && \
    PAPI_VERSION_MINOR(PAPI_VERSION)>=9

#define MAX_COMPONENTS 8

#else

#define MAX_COMPONENTS 1

typedef PAPI_substrate_info_t PAPI_component_info_t;

#define PAPI_COMPONENT_INDEX(a__) 0
#define PAPI_COMPONENT_MASK 

#define PAPI_num_components()  1
#define PAPI_get_component_info(a__) PAPI_get_substrate_info()

#endif


int EventSet[MAX_COMPONENTS];
int NumEvents[MAX_COMPONENTS];

char *xmlize(const char *msg)
{
   char *xmlized_msg, *xp;
   const char *op;

   if(!msg) return NULL;

   /* in the worst case, the string will be 5 times longer, so
    * rather than constantly checking whether we need to realloc,
    * just alloc 5 * strlen(msg) here.
    */

   xmlized_msg = (char *)malloc(5 * strlen(msg));
   if(!xmlized_msg)
     return NULL;

   for(op = msg, xp = xmlized_msg; *op != '\0'; op++) {
       switch(*op) {
         case '"':
           strcpy(xp, "&quot;");
           xp += strlen("&quot;");
           break;
         case '&':
           strcpy(xp, "&amp;");
           xp += strlen("&amp;");
           break;
         case '\'':
           strcpy(xp, "&apos;");
           xp += strlen("&apos;");
           break;
         case '<':
           strcpy(xp, "&lt;");
           xp += strlen("&lt;");
           break;
         case '>':
           strcpy(xp, "&gt;");
           xp += strlen("&gt;");
           break;
         default:
           *xp++ = *op;
       }
   }

   *xp = '\0';

   return xmlized_msg;
}


int papi_xml_hwinfo(FILE *f)
{
  const PAPI_hw_info_t *hwinfo;
  char *xml_string;

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    return(PAPI_ESBSTR);
  
  fprintf(f, "<hardware>\n");
  xml_string = xmlize(hwinfo->vendor_string);
  fprintf(f, "  <vendor string=\"%s\" code=\"%d\"/>\n", 
	  xml_string, hwinfo->vendor);
  free (xml_string);
  xml_string = xmlize(hwinfo->model_string);
  fprintf(f, "  <model string=\"%s\" code=\"%d\"/>\n",
	  hwinfo->model_string, hwinfo->model);
  free (xml_string);
  fprintf(f, "  <system nodes=\"%d\" totalCPUs=\"%d\"/>\n",
	  hwinfo->nnodes, hwinfo->totalcpus);
  fprintf(f, "  <node CPUs=\"%d\"/>\n",
	  hwinfo->ncpu);
  fprintf(f, "  <CPU revision=\"%f\" clockrate=\"%f\" />\n",
	  hwinfo->revision, hwinfo->mhz);
  fprintf(f, "</hardware>\n");

  return (PAPI_OK);
}


void papi_init(int argc, char **argv)
{
  int i;
  int retval;
  
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
    /* fprintf(stdout, "*** Cannot add %d\n", evt);*/
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


void enum_events(FILE *f, int cidx, 
		 const PAPI_component_info_t *comp, 
		 int modifier)
{
  int i, k, num;
  int retval;
  PAPI_event_info_t info;
  char *xml_symbol, *xml_desc;

  i = PAPI_COMPONENT_MASK(cidx)|modifier;

  fprintf(f, "  <eventset type=\"%s\">\n", 
	  modifier&PAPI_PRESET_MASK?"PRESET":"NATIVE" );
  
  num=-1;
  retval=PAPI_enum_event(&i, PAPI_ENUM_FIRST);
  while( retval==PAPI_OK )
    {
      num++;
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
	  xml_symbol = xmlize(info.symbol);
	  xml_desc = xmlize(info.long_descr);
      fprintf(f, "    <event index=\"%d\" name=\"%s\" desc=\"%s\" code=\"0x%x\">\n",
	      num, xml_symbol, xml_desc, info.event_code);
	  free(xml_symbol);
	  free(xml_desc);

      

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

		  xml_symbol = xmlize(strchr(info.symbol, ':'));
		  xml_desc = xmlize(strchr(info.long_descr, ':')+1);
		  fprintf(f, "        <modifier name=\"%s\" desc=\"%s\" code=\"0x%x\"> </modifier>\n",
			  xml_symbol, xml_desc, info.event_code );
		  free(xml_symbol);
		  free(xml_desc);

		}
	      } while (PAPI_enum_event(&k, PAPI_NTV_ENUM_UMASKS) == PAPI_OK);
	    }
	  }
	}

      fprintf(f, "    </event>\n" );

      retval = PAPI_enum_event(&i, PAPI_ENUM_EVENTS);
    }

  fprintf(f, "  </eventset>\n");


  
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
  const PAPI_component_info_t *comp;

  int cidx=-1;
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
	      cidx=(i+1)<argc?atoi(argv[(i++)+1]):-1;
	      if( cidx<0 || cidx>=numc )
		{
		  fprintf(stderr, "Error: component index %d out of bounds (0..%d)\n",
			  cidx, numc-1);
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
  
  if( cidx>=0 )
    {
      comp = PAPI_get_component_info(cidx);

      fprintf(stdout, "<component index=\"%d\" type=\"%s\" id=\"%s\">\n", 
	      cidx, cidx?component_type((char*)(comp->name)):"CPU", comp->name );

      if( native )
	enum_events(stdout, cidx, comp, PAPI_NATIVE_MASK);
      if( preset )
	enum_events(stdout, cidx, comp, PAPI_PRESET_MASK);

      fprintf(stdout, "</component>\n");
    }
  else
    {
      for( cidx=0; cidx<numc; cidx++ )
	{
	  comp = PAPI_get_component_info(cidx);

	  fprintf(stdout, "<component index=\"%d\" type=\"%s\" id=\"%s\">\n", 
		  cidx, cidx?component_type((char*)(comp->name)):"CPU", comp->name );
	  
	  if( native )
	    enum_events(stdout, cidx, comp, PAPI_NATIVE_MASK);
	  if( preset )
	    enum_events(stdout, cidx, comp, PAPI_PRESET_MASK);

	  fprintf(stdout, "</component>\n");

	}
    }
  fprintf(stdout, "</eventinfo>\n");

  return 0;
}
