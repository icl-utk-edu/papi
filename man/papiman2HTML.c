/* papiman2HTML.c -- executable to create HTML manpages for papi */
/* file: papiman2HTML.c
 * program for
 * 
 * PerfAPI Library          
 * 
 * July 1999. 
 * 
 * Cricket Haygood Deane 
 * Innovative Computing Labs 
 * University of Tennessee, Knoxville 
 * deane@cs.utk.edu 
 * 
 */ 

/*
 * This is a lightweight tool to create papi_manpages
 * in standard manpage format using nroff and a specialized
 * format file [ papitmac.an ]. The nroff pages are
 * written to a temp file, reformatted for web page display,
 * then written to the directory specified by argv[1]. 
 * The output filename ends with ".txt", which HTML 
 * interprets as a clear-text display. The unix chmod 
 * command is used to set permissions on the output file 
 * as world-readable. 
 *
 * A sed program [ reformat.sed ] is called to clean-out
 * print-control characters embedded in the nroff output. 
 *
 * Processing continues until argv[ARGC] is done.
 * The -p option is for testing non_papi nroff files
 * (in case you want to make your own manpage!)
 *
 * usage: papiman2HTML targetdir  infilename
 *    or: papiman2HTML targetdir  in1 in2 in3 in4
 *    or: papiman2HTML targetdir  PAPI_*
 *    or: papiman2HTML targetdir  non_papi_file -p
 *
 * where: 
 *        infilename is a file that has been prepared in
 *        standard nroff_input_format (such as PAPI_start)
 *
 *        targetdir is target directory
 *
 * To compile this program:  
 *     unix> make papiman2HTML
 *
 * To run this program:
 *     unix> papiman PAPI_function_name(s)
 *
 * for example:
 *     unix> papiman2HTML ~/www-home/papi_dir  PAPI_start
 *      [will put manpage for PAPI_start in ~/www-home/papi_dir/PAPI_start.txt]
 *
 *     unix> papiman2HTML ~/www-home/papi_dir PAPI_*
 *      [will put all PerfAPI Manpages in ~/www-home/papi_dir]
 *
 * This program will NOT make the target directory for you.
 * If the target directory does not exist, the program will fail.
 *    
 */ 
/*
This is the program file for reformat.sed:
This program strips out nroff format characters.
-------------------------------------------------
#reformat.sed
# command line:
# unix> sed -f reformat.sed inputFile > outputFile

s/.//g
 
-------------------------------------------------
*/

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NUMBER_OF_FUNCTIONS 100
char *papi_function_list[NUMBER_OF_FUNCTIONS] = {
	"papiTemplate",
	"PAPI_accum",
	"PAPI_add_event",
	"PAPI_add_events",
	"PAPI_add_pevent",
	"PAPI_cleanup",
	"PAPI_describe_event",
	"PAPI_get_opt",
	"PAPI_init",
	"PAPI_help",
	"PAPI_list_events",
	"PAPI_num_events",
	"PAPI_overflow",
	"PAPI_perror",
	"PAPI_profil",
	"PAPI_query_event",
	"PAPI_read",
	"PAPI_read_counters",
	"PAPI_rem_event",
	"PAPI_rem_events",
	"PAPI_reset",
        "PAPI_restore",
	"PAPI_save",
	"PAPI_set_domain",
	"PAPI_set_granularity",
	"PAPI_set_opt",
	"PAPI_shutdown",
	"PAPI_start", 
        "PAPI_start_counters",
	"PAPI_state",
	"PAPI_stop",
	"PAPI_stop_counters",
	"PAPI_write"
	};

void copy_for_web (char *outFileName); 
	

int main(int argc, char *argv[]) {

int i,M;
char *output_filename,*targetDir;
int function_index=-1;
int allow_flag=0;
int ARGC;
char *errMsg = "No papi manual entry for ";
char *COMMAND1,*COMMAND2;


if (argc< 3) {
	printf("\n usage:  papiman2HTML targetDir functionName \n\n");
	exit(0);
	}

/* malloc everything here */
COMMAND1=(char *)malloc(200 * sizeof(char));
COMMAND2=(char *)malloc(200 * sizeof(char));
output_filename=(char *)malloc( 128*sizeof(char) );
targetDir=(char *)malloc( 64*sizeof(char) );


/* the allow_flag allows processing of non_papi nroff input */
ARGC=argc;
if( !strcmp(argv[argc-1],"-p") ) {
	ARGC--;
	allow_flag++;
	}

/* make targetDir*/
strcpy(targetDir,argv[1]);
strcat(targetDir,"/");

M=2;

while ( M < ARGC ) {

/* name of output file: targetDir/argv[M].txt */
/* all output files go into the SAME targetDir*/
memset(output_filename,0x00,128);
strcpy(output_filename,targetDir);
strcat(output_filename,argv[M]);
strcat(output_filename,".txt");

/* command to run "nroff -e  papitmac.an argv[M] > temp" */
memset(COMMAND1,0x00,200);
strcpy(COMMAND1,"nroff -e  papitmac.an ");
strcat(COMMAND1,argv[M]);
strcat(COMMAND1," > temp");

/* command to set world readable permissions */
memset(COMMAND2,0x00,200);
strcpy(COMMAND2," chmod 755 ");
strcat(COMMAND2,output_filename);


/* -p option to allow testing of nonPAPIfunction files*/
if(allow_flag) {
	function_index=0;
	}

else { /* identify function_index */
	i=0;
	while(papi_function_list[i]) {
	  if(!strcmp(argv[M],papi_function_list[i])) {
   		function_index=i;
   		break;
   		}
	  i++;
	}/* end while */
}/* end else */


if(function_index<0) {
	printf("\n %s %s.\n",errMsg,argv[1]);
	exit(0);
	}

/* make the temp file with nroff*/
system(COMMAND1);

/* strip out backspaces and extra characters with sed*/
system("sed -f reformat.sed temp > temp2");

/* reformat pages from 66 lines to 64 lines and write
   to output_filename */

   copy_for_web(output_filename);

/* set permissions */
system(COMMAND2);

printf("\n %2d. %s ==>> %s", M-1,argv[M],output_filename); 

M++;
}/* end while (M < ARGC) */

printf("\n\n %s processed %d manpage",__FILE__,M-2);

if(M-2>1) printf("s\n\n");
else      printf(" \n\n");


system("rm -f temp");
system("rm -f temp2");
exit(0);

}/*end main*/


void copy_for_web (char *outFileName) 
{

/* this function takes standard nroff page format 
   of 66 lines and makes it into web friendly 
   page format of 62 lines, by eliminating the
   blank lines numbered 0, 1, 2, and 65 
*/ 

FILE *fp,*fp2;
char *buff;
int i,done;

buff=(char *)malloc(200*sizeof(char));
done=0;
fp=fopen("temp2","r");
fp2=fopen(outFileName,"w");

while(!done) {

/* discard first 3 lines*/
for(i=0;i<3;i++) {
   if(!fgets(buff,200,fp)) {done++;break;} 
   }

/* keep lines #3 - #64 */
for(i=3;i<65;i++) {
   memset(buff, 0x00, 200);
   if(!fgets(buff,200,fp)) {done++;break;} 
   fputs(buff,fp2);
   }

/* discard line #65 */
   if(!fgets(buff,200,fp)) {done++;break;} 

	
}/* end while */

fclose(fp);
fclose(fp2);

return;
}
