//-----------------------------------------------------------------------------
// This is a utility program to read pmid.txt and construct source code for
// a string array to exclude lines beginning with '#' or '+' and include other
// pcp events.
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT "PCPEventList.c"
#define INPUT  "pmid.txt"

int main(int argc, char **argv) {
   FILE* myOut=fopen(OUTPUT, "w");                    // Open the file for output.
   if (myOut == NULL) return(-1);                     // Fail if we cannot.
   FILE* myInp=fopen(INPUT, "r");                     // Open the file for input.
   if (myInp == NULL) return(-1);                     // Fail if we cannot.
   fprintf(myOut, "//-----------------------------------------------------------------------------\n"
   fprintf(myOut, "// This file must be included in linux-pcp.c; for that to happen using the PAPI\n"
   fprintf(myOut, "// install script, it must reside in the same directory as linux-pcp.c.        \n"
   fprintf(myOut, "// If the event names below are available from the installed PCP, they will be \n"
   fprintf(myOut, "// available through PAPI. Events that return strings or anything other than an\n"
   fprintf(myOut, "// int or floating point are ignored. Events with PCP multiplicity (e.g. every \n"
   fprintf(myOut, "// cpu, or every socket, etc) are expanded to an event per instance.           \n"
   fprintf(myOut, "//                                                                             \n"
   fprintf(myOut, "// For developers, we do not recommend modifying this file directly. The file  \n"
   fprintf(myOut, "// is constructed by the code buildPCPEventList.c, which is in the /tests/     \n"
   fprintf(myOut, "// directory for the PCP component. You will find pmid.txt there, it can be    \n"
   fprintf(myOut, "// modified and used to rebuild this list.                                     \n"
   fprintf(myOut, "//                                                                             \n"
   fprintf(myOut, "// See README_AddPCPEvents.txt for more detailed instructions.                 \n"
   fprintf(myOut, "//-----------------------------------------------------------------------------\n"
   fprintf(myOut, "char *PCPEventList[]={\n");        // start the string array.

   char line[1024];
   while(1) {
      int ret=fscanf(myInp, "%1024[^\n]\n", line);    // Note leading white space is skipped automatically by fscanf.
      if (ret == EOF) break;                          // Get out if EOF.
      if (line[0] == '#') continue;                   // skip comments and commented out events. 
      if (line[0] == '+') continue;                   // skip comments and commented out events. 
      char* spc = strchr(line, ' ');                  // Find first space in line.
      if (spc != NULL) spc[0]=0;                      // terminate line there.
      if (strncmp(line, "pcp:::", 6) == 0) {          // If it begins with pcp:::, skip that,
         if (strlen(line) < 10) continue;             // .. skip if too short,  
         fprintf(myOut, "   \"%s\",\n", line+6);      // .. print without it, 
      } else {                                        // Otherwise, 
         if (strlen(line) < 4) continue;              // .. skip if too short, 
         fprintf(myOut, "   \"%s\",\n", line);        // .. Write the whole string out.
      }
   }

   fprintf(myOut, "   \".\"}; // End of List Marker.\n");   // Complete the structure.
   fprintf(stderr, "The file %s has been built in this directory.\n"
                   "You must move it to the parent directory; papi/src/components/pcp/ to\n"
                   "be included. Then PAPI must be rebuilt, to activate these PCP events.\n"
                   "A typical rebuild is completed in papi/src, and looks like this:     \n"
                   "> make clobber                                                       \n"
                   "> ./configure --prefix=$PWD/install --with-components=pcp            \n"
                   "> make && make install                                               \n"
                   "However, your particular install may use additional options.         \n"
   ,OUTPUT);

   return(0);
} // end main.  
