//-----------------------------------------------------------------------------
// This is a utility program to combine newpmid.txt with oldpmid.txt to make
// pmid.txt. Events enabled in either one will be enabled in pmid.txt.  I
// recommend copying or renaming pmid.txt to oldpmid.txt before you begin;
// oldpmid.txt will not be changed. Anything commented OUT in oldpmid.txt will
// be discarded, anything commented out in newpmid.txt will be shown as such. 
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT "pmid.txt"
#define INOLD  "oldpmid.txt"
#define INNEW  "newpmid.txt"

int main(int argc, char **argv) {
   int i, j, enableCount=0;
   char line[1024], ename[1024];

   FILE* myOut=fopen(OUTPUT, "w");              // Open the file for output.
   if (myOut == NULL) return(-1);               // Fail if we cannot.
   FILE* myOLD=fopen(INOLD, "r");               // Open the file for input.
   FILE* myNEW=fopen(INNEW, "r");               // Open the file for input.
   
   if (myOLD == NULL || myNEW == NULL) {        // If either failed,
      fprintf(stderr, "This program requires files '%s' and '%s' already exist in the current directory.\n", INOLD, INNEW);
      exit(-1);
   }

   // Not doing anything fancy. First read the old file, to find how many events are in it.
   int oldCount = 0;
   while(1) {
      int ret=fscanf(myOLD, "%1024[^\n]\n", line);    // Read a line.
      if (ret == EOF) break;                          // Exit with count.
      if (line[0] != '#') oldCount++;                 // count a valid line.
   }

   char **oldEvents=malloc(oldCount*sizeof(char*));   // My array of old events.
   rewind(myOLD);                                     // Start from the beginning.

   // Reread and store all events.
   oldCount=0;                                        // reset counter.
   while(1) {
      int ret=fscanf(myOLD, "%1024[^\n]\n", line);    // Read a line.
      if (ret == EOF) break;                          // Exit with count.
      if (line[0] == '#') continue;                   // Skip a comment.
      ret = sscanf(line, "%1024[^ ]", ename);         // Try to read the name. 
      oldEvents[oldCount]=calloc(1024, sizeof(char)); // Make some space.
      strncpy(oldEvents[oldCount++], ename, 1024);    // Copy the event over, increment the index.
   }

   // Now read the NEW file. Any lines NOT beginning
   // with '#' are moved directly. Those beginning 
   // with '#' are rescanned to collect the event 
   // name, if we find that in the oldEvents, we 
   // uncomment the line before writing it out.
   // We also 'flag' the event found; by changing
   // the first character to 0; so we can report
   // old events we did NOT find.

   while(1) {
      int ret=fscanf(myNEW, "%1024[^\n]\n", line);    // Note leading white space is skipped automatically by fscanf.
      if (ret == EOF) break;                          // Get out if EOF.
      if (line[0] != '#') {                           // If an enabled event,
         fprintf(myOut, "  %s\n", line);              // .. duplicate it.
         enableCount++;                               // .. For a report.
         continue;                                    // .. done with this line.
      }

      // Here we know the line is commented out.
      ret = sscanf(line, "# %[^ ]", ename);           // Collect the event name for lookup.
      for (i=0; i<oldCount; i++) {
         if (strcmp(oldEvents[i], ename) == 0) break; // Break if we find it.
      }

      if (i<oldCount) {                               // If we did find it,
         fprintf(stderr,"Enabling: %s\n", line);      // .. Show if we enable something previously commented out.
         line[0]=' ';                                 // .. remove comment mark.
         oldEvents[i][0] = 0;                         // .. mark this one read.
         enableCount++;                               // .. remember we output an enabled event.
      }

      fprintf(myOut,"%s\n", line);                    // Write it out, either way.
   }

   fprintf(stderr, "Total Enabled Events: %i\n", enableCount);
   j = 0;                                             // init count of old events not found in new file.
   for (i=0; i<oldCount; i++) {
      if (oldEvents[i][0] != 0) j++;                  // count it.
   }

   if (j == 0) {                                      // If all accounted for,
      fprintf(stderr,"All events in oldpmid.txt were enabled in pmid.txt.\n");
   } else {
      fprintf(stderr,"The following events (%i) in oldpmid.txt were not found in newpmid.txt:\n", j);
      for (i=0; i<oldCount; i++) {
         if (oldEvents[i][0] != 0) fprintf(stderr, "%s\n", oldEvents[i]);
      }
   }
   
   for (i=0; i<oldCount; i++) {                       // For each old event,
      free(oldEvents[i]);                             // ..free malloced memory.
   }

   free(oldEvents);                                   // Free the pointer array.
   fclose(myOut);
   fclose(myOLD);
   fclose(myNEW);

   return(0);
} // end main. 
