#include <windows.h>
#include <stdio.h>

void CPAUSE(void)
{
   HANDLE hStdIn;
   BOOL bSuccess;
   INPUT_RECORD inputBuffer;
   DWORD dwInputEvents;         /* number of events actually read */

   printf("Press any key to continue...\n");
   hStdIn = GetStdHandle(STD_INPUT_HANDLE);
   do {
      bSuccess = ReadConsoleInput(hStdIn, &inputBuffer, 1, &dwInputEvents);
   } while (!(inputBuffer.EventType == KEY_EVENT && inputBuffer.Event.KeyEvent.bKeyDown));
}
