// WinPAPIShell.c : Defines the entry point for the application.
//
#include "stdafx.h"
#include <winioctl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <stdio.h>
#include "resource.h"
#include "winpmc.h"

// Global Variables:
HINSTANCE hInst;								// current instance
TCHAR appDir[256];								// application directory
TCHAR helpDir[256];								// help file directory

// Foward declarations of functions included in this code module:
BOOL				InitInstance(HINSTANCE, int);
LRESULT CALLBACK	About(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK	getFileHook(HWND, UINT, WPARAM, LPARAM);
static void			openPAPItest(void);
static void			exerciseDriver(void);
static void			centerDialog(HWND hdlg);
static void			doHelp(void);


int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow)
{

	// Perform application initialization:
	if (!InitInstance (hInstance, nCmdShow)) return FALSE;

	// The About 
	DialogBox(hInst, (LPCTSTR)IDD_ABOUTBOX1, NULL, (DLGPROC)About);
	return(WM_QUIT);
}


//
//   FUNCTION: InitInstance(HANDLE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   GetCurrentDirectory(sizeof(appDir), appDir);

   return TRUE;
}


// test if directory path exists
static BOOL findPath(char *path) {
	WIN32_FIND_DATA findData;
	HANDLE findHandle;


	findHandle = FindFirstFile(path, &findData);
	if (findHandle != INVALID_HANDLE_VALUE) {
		FindClose(findHandle);
		return (TRUE);
	}
	return(FALSE);
}

static int stripDir(char *dir) {
	int i;
	for (i=strlen(dir)-1; i>1; i--)
		if (dir[i] == '\\') break;
	if (i > 1) dir[i] = 0;
	return (i);
}


static void enableHelp(HWND hDlg) {
	HWND itemHndl;
	int i = 2;
	char help[] = "\\help\\welcome.html";
	char man[]  = "\\man\\html\\papi.html";

	itemHndl = GetDlgItem(hDlg, IDHELP);

	strcpy(helpDir, appDir);
	strcat(helpDir, help);

	// look for \help inside default directory
	if (!findPath(helpDir)) {
		// look for /help up one level
		strcpy(helpDir, appDir);
		i = stripDir(helpDir);
		if (i > 1) {
			strcat(helpDir, help);
			if (!findPath(helpDir)) {
				// look for help in /man/html
				strcpy(helpDir, appDir);
				i = stripDir(helpDir);
				i = stripDir(helpDir);
				i = stripDir(helpDir);
				if (i > 1) {
					strcat(helpDir, man);
					if (!findPath(helpDir)) i = 0;
				}
			}
		}	
	}
	if (i < 2) EnableWindow(itemHndl, 0);
}


static void enableDriver(HWND hDlg) {
	HWND itemHndl;
	char fname[256];

	GetSystemDirectory(fname, sizeof(fname));
	strcat(fname, "\\drivers\\winpmc.sys");

	itemHndl = GetDlgItem(hDlg, IDDRIVER);

	// look for driver file
	if (!findPath(fname)) EnableWindow(itemHndl, 0);
}


// Mesage handler for about box, which serves as the main interface
LRESULT CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		int wmId, wmEvent;

		case WM_INITDIALOG:
			centerDialog(hDlg);
			enableHelp(hDlg);
			enableDriver(hDlg);
			return TRUE;

		case WM_COMMAND:
			wmId    = LOWORD(wParam); 
			wmEvent = HIWORD(wParam); 
			// Parse the menu selections:
			switch (wmId)
			{
 				case IDM_PAPI_TESTS:
 				case IDTEST:
					openPAPItest();
					return TRUE;

                case IDM_TEST_KERNEL:
                case IDDRIVER:
					exerciseDriver();
					return TRUE;

				case IDHELP:
					doHelp();
					return TRUE;

				case IDWEB:
					ShellExecute(NULL,"open","http://icl.cs.utk.edu/projects/papi", NULL, NULL, SW_MAXIMIZE);
					return TRUE;

				case IDOK:
				case IDCANCEL:
					EndDialog(hDlg, LOWORD(wParam));
					return TRUE;
			}
			break;
	}
    return FALSE;
}

// find and open a PAPI test application in a console window
static void openPAPItest(void)
{
	HINSTANCE myInst;
	OPENFILENAME ofn;
	BOOL gotFile;
	char testDir[256];
	char *filter = "PAPI Test Executables\0*.exe\0\0";
	char filename[256] = "\0";
	char *defExt = "exe";
	char *Title = "PAPI Test Application Selection";

	memset(&ofn,0,sizeof(OPENFILENAME));

	// default to application directory
	ofn.lpstrInitialDir = appDir;

	strcpy(testDir, appDir);
	strcat(testDir, "\\tests");
	// look for /tests inside default directory
	if (findPath(testDir)) ofn.lpstrInitialDir = testDir;
	else {
		int i;
		// look for /tests up one level
		strcpy(testDir, appDir);
		i = stripDir(helpDir);
		if (i > 1) {
			strcat(testDir, "\\tests");
			if (findPath(testDir)) ofn.lpstrInitialDir = testDir;
		}	
	}

	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.lpstrFilter = filter; 
	ofn.lpstrFile = filename; 
	ofn.nMaxFile = sizeof(filename); 
	ofn.lpstrTitle = Title;
	ofn.lpstrDefExt = defExt;
	ofn.Flags = OFN_FILEMUSTEXIST |	OFN_PATHMUSTEXIST 
			  | OFN_HIDEREADONLY | OFN_ENABLEHOOK | OFN_EXPLORER;
	ofn.lpfnHook = getFileHook;
	gotFile = GetOpenFileName(&ofn);
	if (gotFile)
		myInst = ShellExecute(NULL,"open",filename, NULL, NULL, SW_MAXIMIZE);
}

// put the driver through its paces to make sure it's there and active
static void exerciseDriver(void)
{    
	HANDLE hDriver = INVALID_HANDLE_VALUE;
	DWORD dwBytesReturned;
	BOOL  bReturnCode = FALSE;
	char szString[256]; // character buffer
	int iobuf[256];     // I/O buffer

	// Try opening a static device driver. 
	hDriver = CreateFile("\\\\.\\WinPMC",
			 GENERIC_READ | GENERIC_WRITE, 
			 FILE_SHARE_READ | FILE_SHARE_WRITE,
			 0,                     // Default security
			 OPEN_EXISTING,
			 0,						// Don't Perform asynchronous I/O
			 0);                    // No template

	if (hDriver == INVALID_HANDLE_VALUE)
			MessageBox(NULL,"Bummer","Driver Load Failed.",MB_OK);
	else {

		// Dispatch the READ_TEST_STRING IOCTL to our NT driver.
		bReturnCode = DeviceIoControl(hDriver,
					  IOCTL_PMC_READ_TEST_STRING,
					  NULL, 0, iobuf, sizeof(iobuf),
					  &dwBytesReturned, NULL);

		// Display the results!
		MessageBox(NULL,(const char *)iobuf,"WinPMC Welcome",MB_OK);
/*
		// Send a request to the driver. The request code is HELLO, no parameters
		bReturnCode = DeviceIoControl(hDriver, HELLO, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
		if (bReturnCode) {
		  sprintf(szString, "HELLO RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
		  	MessageBox(NULL, szString, "HELLO Test",MB_OK);
		}
		else 	MessageBox(NULL,"HELLO failed.","HELLO Test",MB_OK);


		// Send a request to the driver. The request code is HELLONUM, one integer parameter
		iobuf[0] = 319;    // my favorite number :-)
		bReturnCode = DeviceIoControl(hDriver, HELLONUM, iobuf, sizeof(int), iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
		if (bReturnCode) {
		  sprintf(szString, "HELLONUM RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
		  	MessageBox(NULL,szString, "HELLONUM Test",MB_OK);
		}
		else 	MessageBox(NULL,"HELLONUM failed.","HELLONUM Test",MB_OK);
*/
		// Send a request to the driver. The request code is TASKSWITCH, no parameters

		bReturnCode = DeviceIoControl(hDriver, TASKSWITCH, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
		if (bReturnCode) {
			if (iobuf[0] == 0) {
				MessageBox(NULL, "This machine is running the Uniprocessor Free Build.","TASKSWITCH Test",MB_OK);
				__asm mov ecx, 0x00000000
				__asm rdpmc
				MessageBox(NULL,"We have successfully executed RDPMC from user space!","RDPMC Test...",MB_OK);
			}
			else {
				strcpy(szString, "This machine is running the Multiprocessor or Checked Build.");
				strcat(szString, "\n It cannot currently support PAPI.");
				sprintf(&szString[strlen(szString)], "\nThere have been %d task switches since the driver was opened.", iobuf[0]);
		  		MessageBox(NULL, szString, "TASKSWITCH Test",MB_OK);
			}
		}
		else MessageBox(NULL,"TASKSWITCH failed.","TASKSWITCH Test",MB_OK);

		CloseHandle(hDriver);
	}
}



static void doHelp(void)
{
	HINSTANCE myInst;
	myInst = ShellExecute(NULL,"open", helpDir, NULL, NULL, SW_SHOWNORMAL);
}


LRESULT CALLBACK getFileHook(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (message == WM_INITDIALOG) centerDialog(GetParent(hDlg));
	return(FALSE);
}

//______________________________________________________________________________________
static void centerDialog(HWND hdlg)
{
	RECT r;
	short left, top, screenWd, screenHt;

	// center a dialog on the screen
	GetWindowRect(hdlg, &r);
	screenWd = GetSystemMetrics(SM_CXSCREEN);
	screenHt = GetSystemMetrics(SM_CYSCREEN);
	left = (screenWd - (r.right  - r.left)) / 2;
	top  = (screenHt - (r.bottom - r.top)) / 3;
	SetWindowPos(hdlg, NULL, left, top, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
}

