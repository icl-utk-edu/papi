// WinPAPI_console.c : Defines routines to handle a dos console.
//

#include "stdafx.h"
#include "io.h"
#include "fcntl.h"
#include "stdio.h"

// routine to create a printf console and set up the standard handles
void enterConsole(LPSTR title, short nChars, short nLines)
{
	HANDLE hConsole;
	int hCrt;
	FILE *hf;
	long lastError;

	AllocConsole();
	SetConsoleTitle(title);

	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	hCrt = _open_osfhandle((long)hConsole,_O_TEXT);
	hf = _fdopen(hCrt, "w");
	*stdout = *hf;
	setvbuf(stdout, NULL, _IONBF, 0);

	hCrt = _open_osfhandle((long)GetStdHandle(STD_ERROR_HANDLE),
		_O_TEXT);
	hf = _fdopen(hCrt, "w");
	*stderr = *hf;
	setvbuf(stderr, NULL, _IONBF, 0);

	lastError = resizeConBufAndWindow(hConsole, nChars, nLines);
}

// routine to wait for a keypress and exit the console window
void exitConsole(void)
{
	waitConsole();
	FreeConsole();
}


// routine to wait for a keypress and exit the console window
void waitConsole(void)
{
	HANDLE hStdIn;
	BOOL bSuccess;
	INPUT_RECORD inputBuffer;
	DWORD dwInputEvents; /* number of events actually read */

	printf("Press any key to continue...\n");
	hStdIn = GetStdHandle(STD_INPUT_HANDLE);
	do { bSuccess = ReadConsoleInput(hStdIn, &inputBuffer, 
		1, &dwInputEvents);
	} while (!(inputBuffer.EventType == KEY_EVENT &&
		inputBuffer.Event.KeyEvent.bKeyDown));
}



/******************************************************************************\
*       This is lifted from the Microsoft Source Code Samples. 
*       Copyright (C) 1993-1997 Microsoft Corporation.
*       All rights reserved. 
\******************************************************************************/


/*********************************************************************
* FUNCTION: resizeConBufAndWindow(HANDLE hConsole, SHORT xSize,      *
*                                 SHORT ySize)                       *
*                                                                    *
* PURPOSE: resize both the console output buffer and the console     *
*          window to the given x and y size parameters               *
*                                                                    *
* INPUT: the console output handle to resize, and the required x and *
*        y size to resize the buffer and window to.                  *
*                                                                    *
* COMMENTS: Note that care must be taken to resize the correct item  *
*           first; you cannot have a console buffer that is smaller  *
*           than the console window.                                 *
* RETURNS:  0 if successful, or GetLastError() code if not.          *
*********************************************************************/

DWORD resizeConBufAndWindow(HANDLE hConsole, SHORT xSize, SHORT ySize)
{
  CONSOLE_SCREEN_BUFFER_INFO csbi; /* hold current console buffer info */
  SMALL_RECT srWindowRect; /* hold the new console size */
  COORD coordScreen;

  if (!GetConsoleScreenBufferInfo(hConsole, &csbi)) return(GetLastError());

  /* get the largest size we can size the console window to */
  coordScreen = GetLargestConsoleWindowSize(hConsole);

  /* define the new console window size and scroll position */
  srWindowRect.Right = (SHORT) (min(xSize, coordScreen.X) - 1);
  srWindowRect.Bottom = (SHORT) (min(ySize, coordScreen.Y) - 1);
  srWindowRect.Left = srWindowRect.Top = (SHORT) 0;
 
  /* define the new console buffer size */
  coordScreen.X = xSize;
  coordScreen.Y = ySize;
 
  /* if the current buffer is larger than what we want, resize the */
  /* console window first, then the buffer */
  if ((DWORD) csbi.dwSize.X * csbi.dwSize.Y > (DWORD) xSize * ySize)
    {
    if (!SetConsoleWindowInfo(hConsole, TRUE, &srWindowRect)) return(GetLastError());

    if (!SetConsoleScreenBufferSize(hConsole, coordScreen)) return(GetLastError());

  }
  /* if the current buffer is smaller than what we want, resize the */
  /* buffer first, then the console window */
  if ((DWORD) csbi.dwSize.X * csbi.dwSize.Y < (DWORD) xSize * ySize)
    {
    if (!SetConsoleScreenBufferSize(hConsole, coordScreen)) return(GetLastError());
    if (!SetConsoleWindowInfo(hConsole, TRUE, &srWindowRect)) return(GetLastError());
    }
  /* if the current buffer *is* the size we want, don't do anything! */
  return(0);
}

