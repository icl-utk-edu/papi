// WinPAPIShell.c : Defines the entry point for the application.
//
#include "stdafx.h"
#include <winioctl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <stdio.h>
#include "resource.h"
#include "papi.h"
#include "winpmc.h"


// Global Variables:
HINSTANCE hInst;		// current instance
TCHAR appDir[256];		// application directory
TCHAR CDir[256];		// C example directory
TCHAR FortranDir[256];	// Fortran example directory
TCHAR PerfDir[256];		// Perfometer example directory
TCHAR jarDir[256];		// Perfometer jar application directory
TCHAR helpDir[256];		// help file directory

// Foward declarations of functions included in this code module:
BOOL			InitInstance(HINSTANCE, int);
LRESULT CALLBACK	About(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK	Diagnostics(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK	getFileHook(HWND, UINT, WPARAM, LPARAM);
static BOOL		UniProcessorBuild(void);
static void		exerciseDriver(void);
static void		getDriverVersion(char *, int);
static void		showDriverVersion(void);
static void		centerDialog(HWND hdlg);
static void		centerDiagnostics(HWND hdlg);
static void		DiagRDPMC(void);
static void		TaskSwitchTest(void);
static void		HelloNumTest(void);
static void		HelloTest(void);
static void		DoLoop(void);

// support routines for exercising counters
static void do_flops(int n);
static unsigned __int64 read_pmc(int reg);


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

// strip the last directory from a path
// up to and including the backslash
// return the resulting string length
static int stripDir(char *dir) {
	int i;
	for (i=strlen(dir)-1; i>1; i--)
		if (dir[i] == '\\') break;
	if (i > 1) dir[i] = 0;
	return (i);
}

// strip the last 'count' directories from a path
// by calling stripDir()
static int stripDirs(char *dir, int count) {
	int i, j;
	for (i=0; i<count; i++) {
		j = stripDir(dir);
		if (j == 0) break;
	}
	return (j);
}


// looks for the example executables where we expect them to live
// pass in a default install directory path and a development path
// the default is relative to the application
// the development path is assumed to be 'stripCnt' directories above the app
static void initPath(char *defPath, char *devPath, char *destPath, int stripCnt) {
	int i = 2;

	strcpy(destPath, appDir);
	strcat(destPath, defPath);
	// look inside default install directory
	if (!findPath(destPath)) {
		// look in the development directory
		strcpy(destPath, appDir);
		i = stripDirs(destPath, stripCnt);
		if (i > 1) {
			strcat(destPath, devPath);
			if (!findPath(destPath)) i = 0;
		}
	}
	// if we can't find it, clear the directory entry
	// let the user look the first time he asks
	if (i < 2) destPath[0] = 0;
}


// search for the existence of PMC Driver
// in the \system32\drivers directory
static BOOL findDriver() {
	char fname[256];

	GetSystemDirectory(fname, sizeof(fname));
	strcat(fname, "\\drivers\\winpmc.sys");

	// look for driver file
	return(findPath(fname));
}


// initialize paths for all buttons
// disable help button if path not found
// disable example buttons if driver not found
static void enableButtons(HWND hDlg) {
	HWND itemHndl;

	// enable the help button iff we find a path
	initPath("\\help\\welcome.html", "\\man\\html\\papi.html", helpDir, 3);
	if (!strlen(helpDir)) {
		itemHndl = GetDlgItem(hDlg, IDHELP);
		EnableWindow(itemHndl, 0);
	}
	// if we don't have the driver, we shouldn't run any examples
	if (!findDriver()) {
		itemHndl = GetDlgItem(hDlg, IDCEX);
		EnableWindow(itemHndl, 0);
		itemHndl = GetDlgItem(hDlg, IDFORTRANEX);
		EnableWindow(itemHndl, 0);
		itemHndl = GetDlgItem(hDlg, IDPERFOMETEREX);
		EnableWindow(itemHndl, 0);
	}
	// initialize the example directory paths
	else {
		initPath("\\ctests", "\\ctests\\Release", CDir, 1);
		initPath("\\ftests", "\\ftests\\Release", FortranDir, 1);
		initPath("\\perfometer", "\\tools\\perfometer\\tests\\Release", PerfDir, 3);
	}
	// initialize the path to the perfometer app
	initPath("\\perfometer.jar", "\\tools\\perfGUI\\perfometer.jar", jarDir, 3);
}


// find and open a PAPI example application in a console window
// if the preinitialized directory is empty
// open the application directory and ask for help
static void openShell(char *filter, char *title, char *ext, char *defDir)
{
	HINSTANCE myInst;
	OPENFILENAME ofn;
	BOOL gotFile;
	char filename[256] = "\0";

	memset(&ofn,0,sizeof(OPENFILENAME));

	ofn.lpstrInitialDir = defDir;
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.lpstrFilter = filter; 
	ofn.lpstrFile = filename; 
	ofn.nMaxFile = sizeof(filename); 
	ofn.lpstrTitle = title;
	ofn.lpstrDefExt = ext;
	ofn.Flags = OFN_FILEMUSTEXIST |	OFN_PATHMUSTEXIST 
			  | OFN_HIDEREADONLY | OFN_ENABLEHOOK | OFN_EXPLORER;
	ofn.lpfnHook = getFileHook;
	gotFile = GetOpenFileName(&ofn);
	if (gotFile)
		myInst = ShellExecute(NULL,"open",filename, NULL, NULL, SW_MAXIMIZE);
}


// find and open the PAPI Perfometer GUI application
static void openPerfometer(void)
{
	char *filter = "Perfometer GUI (*.jar)\0*.jar\0\0";
	char *title  = "Please help me find the Perfometer GUI";
	char *defExt = "jar";

	if (jarDir) ShellExecute(NULL,"open", jarDir, NULL, NULL, SW_SHOWNORMAL);
	else openShell(filter, title, defExt, appDir);
}
	

// find and open a PAPI example application in a console window
// if the preinitialized directory is empty
// open the application directory and ask for help
static void openExamples(char *title, char *dir)
{
	int i,j;
	char fullFilter[128];
	char fullTitle[128];
	char *filterStub = "\0*.exe\0\0";
	char *titleStub = " Selection";
	char *defExt = "exe";
	char *defDir;

	// build filter and title strings
	strcpy(fullTitle, title);
	strcat(fullTitle, titleStub);

	strcpy(fullFilter, title);
	strcat(fullFilter, " Executables (*.exe)");
	j = strlen(fullFilter);
	for (i=0;i<8;i++) fullFilter[i+j] = filterStub[i];

	// set up default directory
	if (strlen(dir)) defDir = dir;
	else {
		strcpy(fullTitle, "Please help me find ");
		strcat(fullTitle, title);
		if (fullTitle[strlen(fullTitle)-1] != 's');
			strcat(fullTitle, "s");
		defDir = appDir;
	}
	openShell(fullFilter, fullTitle, defExt, defDir);
}

// really should add error checking to the file operations...
static void addline(const char *dir, const char *name, FILE *file)
{
	int NumWritten;

	NumWritten = fwrite(name, sizeof(char), strlen(name), file);
	NumWritten = fwrite(" TESTS_QUIET\n", sizeof(char), 13, file);
}

// really should add error checking to the file operations...
static void make_smoke(const char *dir, FILE *out)
{
	HANDLE findFile;
    WIN32_FIND_DATA FindFileData; 	// pointer to returned information 
	char wildcard[] = "\\*.exe";
	char findname[256];

	strcpy(findname, "CD ");
	strcat(findname, dir);
	strcat(findname, "\n");
	fwrite(findname, sizeof(char), strlen(findname), out);

	strcpy(findname, dir);
	strcat(findname, wildcard);
	findFile = FindFirstFile(findname, &FindFileData);
	if (findFile != INVALID_HANDLE_VALUE) {
		addline(dir, FindFileData.cFileName, out);
		while (FindNextFile(findFile, &FindFileData))
			addline(dir, FindFileData.cFileName, out);
		FindClose(findFile);
	}
}
				
// test for the uniprocessor build & fail if not present
// otherwise, build .bat files dynamically to execute all tests
// really should add error checking to the file operations...
static void smokeTest(void)
{
	FILE *batfile;
	TCHAR smokebat[] = "smoke_test.bat";	// file name for batch file
	TCHAR currentdir[256];					// name of current directory

	if (UniProcessorBuild()) {
		// find and run all the C and Fortran tests
		if (strlen(FortranDir) || strlen(CDir)) {
			batfile = fopen(smokebat, "w");
			fwrite("ECHO OFF\n", sizeof(char), 9, batfile);
			if (strlen(FortranDir)) {
				make_smoke(FortranDir, batfile);
			}
			if (strlen(CDir)) make_smoke(CDir, batfile);
			fwrite("PAUSE\n", sizeof(char), 6, batfile);
			fclose(batfile);

			GetCurrentDirectory(sizeof(currentdir), currentdir);
			ShellExecute(NULL, NULL, smokebat, NULL, NULL, SW_MAXIMIZE);
			SetCurrentDirectory(currentdir);
		}
		else MessageBox(NULL, "The low-level driver looks ok, \nbut I couldn't find any test directories.", "Smoke Test",MB_OK);
	}
	else MessageBox(NULL, "This version of PAPI doesn't run on the MultiProcessor Build.", "Smoke Test",MB_OK);
}


static void dispVersions(HWND hDlg)
{
  unsigned int len;
  char text[512] = {"Fourscore and seven years ago..."};
  LPARAM lParam = (LPARAM)text;

  sprintf(text, "   PAPI Library Version: %d.%d.%d;  ",
    PAPI_VERSION_MAJOR(PAPI_VERSION), 
    PAPI_VERSION_MINOR(PAPI_VERSION),
    PAPI_VERSION_REVISION(PAPI_VERSION));
  len = strlen(text);
  getDriverVersion(&text[len], sizeof(text) - len);
  SendDlgItemMessage(hDlg, IDC_STATIC, WM_SETTEXT, 0, lParam);
}


static HWND diagDlg;

// Message handler for about box, which serves as the main interface
LRESULT CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
	int wmId, wmEvent;

	case WM_INITDIALOG:
	    centerDialog(hDlg);
	    dispVersions(hDlg);
	    enableButtons(hDlg);
	    diagDlg = 0;
	    return TRUE;

	case WM_COMMAND:
	    wmId    = LOWORD(wParam); 
	    wmEvent = HIWORD(wParam); 
	    // Parse the menu selections:
	    switch (wmId)
	    {
 		case IDCEX:
		    openExamples("PAPI C Example", CDir);
		    return TRUE;

 		case IDFORTRANEX:
		    openExamples("PAPI Fortran Example", FortranDir);
		    return TRUE;

 		case IDPERFOMETER:
		    openPerfometer();
		    return TRUE;

 		case IDPERFOMETEREX:
		    openExamples("Perfometer Example", PerfDir);
		    return TRUE;

		case IDDIAGNOSTIC:
		    DialogBox(hInst, (LPCTSTR)IDD_ABOUTBOX2, NULL, (DLGPROC)Diagnostics);
		    return TRUE;

		case IDHELP:
		    ShellExecute(NULL,"open", helpDir, NULL, NULL, SW_SHOWNORMAL);
		    return TRUE;

		case IDWEB:
		    ShellExecute(NULL,"open","http://icl.cs.utk.edu/projects/papi", NULL, NULL, SW_MAXIMIZE);
		    return TRUE;

		case IDOK:
		case IDCANCEL:
		    if (diagDlg) 
		      EndDialog(diagDlg, LOWORD(wParam));
		    EndDialog(hDlg, LOWORD(wParam));
		    return TRUE;
	    }
	    break;
    }
    return FALSE;
}


// Message handler for the diagnostics box, which chains from the main about box
LRESULT CALLBACK Diagnostics(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
	int wmId, wmEvent;

	case WM_INITDIALOG:
	    centerDiagnostics(hDlg);
	    diagDlg = hDlg;
	    return TRUE;

	case WM_COMMAND:
	    wmId    = LOWORD(wParam); 
	    wmEvent = HIWORD(wParam); 
	    // Parse the menu selections:
	    switch (wmId)
	    {
 		case IDRDPMC:
		  DiagRDPMC();
		  return TRUE;

		case IDSMOKE:
		  smokeTest();
		  return TRUE;

		case IDVERSION:
		  showDriverVersion();
		  return TRUE;

		case IDTASKSWITCH:
		  TaskSwitchTest();
		  return TRUE;

		case IDDOLOOP:
		  DoLoop();
		  return TRUE;

		case IDHELLO:
		  HelloTest();
		  return TRUE;

		case IDHELLONUM:
		  HelloNumTest();
		  return TRUE;

		case IDHELP:
		  ShellExecute(NULL,"open", helpDir, NULL, NULL, SW_SHOWNORMAL);
		  return TRUE;

		case IDWEB:
		  ShellExecute(NULL,"open","http://icl.cs.utk.edu/projects/papi", NULL, NULL, SW_MAXIMIZE);
		  return TRUE;

		case IDOK:
		case IDCANCEL:
		  diagDlg = 0;
		  EndDialog(hDlg, LOWORD(wParam));
		  return TRUE;
	    }
	    break;
    }
    return FALSE;
}


// put the driver through its paces to make sure it's there and active
// check the task switch stuff to make sure we're using UniProcessor Build
// return TRUE if UniProcessor and everything worked
// return FALSE with a dialog box if anything fails
static BOOL UniProcessorBuild(void)
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
		MessageBox(NULL,"Bummer...","PMC Kernel Driver Load Failed.",MB_OK);
	else {
		// Send a request to the driver. The request code is TASKSWITCH, no parameters
		bReturnCode = DeviceIoControl(hDriver, TASKSWITCH, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
		if (bReturnCode) {
			if (iobuf[0] != 0) {
				strcpy(szString, "This machine is running the Multiprocessor or Checked Build.");
				strcat(szString, "\n It cannot currently support PAPI.");
				sprintf(&szString[strlen(szString)], "\nThere have been %d task switches since the driver was opened.", iobuf[0]);
		  		MessageBox(NULL, szString, "TASKSWITCH Test",MB_OK);
			}
			bReturnCode = TRUE;
		}
		else MessageBox(NULL,"TASKSWITCH failed.","TASKSWITCH Test",MB_OK);

		CloseHandle(hDriver);
	}
	return(bReturnCode);
}


// put the driver through its paces to make sure it's there and active
static HANDLE LoadDriver(void)
{    
    HANDLE hDriver = INVALID_HANDLE_VALUE;

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
    return (hDriver);
}


// get the driver version string; also makes sure it's there and active
static void getDriverVersion(char *version, int size)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {

	// Dispatch the PMC_VERSION_STRING IOCTL to our NT driver.
	bReturnCode = DeviceIoControl(hDriver,
				  IOCTL_PMC_VERSION_STRING,
				  NULL, 0, (int *)version, size,
				  &dwBytesReturned, NULL);
	CloseHandle(hDriver);
	version[dwBytesReturned-2] = 0; // make sure it's terminated
    } else version[0] = 0;
}

// show the driver version string
static void showDriverVersion(void)
{    
    char version[512];     // version string buffer

    getDriverVersion(version, sizeof(version));

    if (strlen(version))
	// Display the results!
	MessageBox(NULL,(const char *)version,"WinPMC Version",MB_OK);
}


// put the driver through its paces to make sure it's there and active
static void DiagRDPMC(void)
{    
    HANDLE hDriver;

    // Try opening a static device driver. 
    hDriver = LoadDriver();
    if (hDriver != (INVALID_HANDLE_VALUE)) {
	MessageBox(NULL,"Ready to execute RDPMC from user space!","RDPMC Test...",MB_OK);
	__asm mov ecx, 0x00000000
	__asm rdpmc
	MessageBox(NULL,"We have successfully executed RDPMC from user space!","RDPMC Test...",MB_OK);
	CloseHandle(hDriver);
    }
}


// set the counters, do some work, and report the results
static void DoLoop(void)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    char szString[256]; // character buffer
    int iobuf[256];     // I/O buffer

	unsigned int evntsel[nPMC] = {0}; // raw events to program into the counters
	unsigned __int64 pmc[nPMC];	/* pmc[0], ..., pmc[n] */
	struct pmc_info *info;	// a pointer to the processor info
	int i, n;

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {
		// Get some processor info from the driver with an IOCTL_PMC_INFO call
		bReturnCode = DeviceIoControl(hDriver, IOCTL_PMC_INFO, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
		if (bReturnCode) {
			info = (struct pmc_info *)iobuf;
			sprintf(szString, "IOCTL_PMC_INFO succeeded;\nRETURNING %d bytes\n", dwBytesReturned);
			strcat(szString, "\nVendor  : ");
			strncat(szString, info->vendor, 12);
			sprintf(szString + strlen(szString), "\nFamily  : %d\nModel   : %d\nStepping: %d\nFeatures: 0x%8x\n", 
			    info->family, info->model, info->stepping, info->features);
			if (info->family == 6) {
			    MessageBox(NULL, szString, "DoLoops",MB_OK);
			    evntsel[0] = 0x4100c1;	// floating point operations (Pentium M)
			    evntsel[1] = 0x010079;	// cycles (Pentium M)
				
			}
			else if (info->family == 15 && !strncmp(info->vendor, "AuthenticAMD",12)) {
			    MessageBox(NULL, szString, "DoLoops",MB_OK);
			    evntsel[0] = 0x410300;	// floating point operations (Opteron)
			    evntsel[1] = 0x410076;	// cycles (Opteron)
			    evntsel[2] = 0x010000;	// NOP
			    evntsel[3] = 0x010000;	// NOP
			}
			else {
			    strcat(szString, "Sorry, this test only works on Pentium III and Opteron class cpus");
			    MessageBox(NULL, szString, "DoLoops",MB_OK);
			    CloseHandle(hDriver);
			    return;
			}
		}
		else {
		    bReturnCode = GetLastError();
		    sprintf(szString, "IOCTL_PMC_INFO Failed\nLast Error Code: %d\n", bReturnCode);
		    MessageBox(NULL, szString, "DoLoops", MB_OK);
		    CloseHandle(hDriver);
		    return;
		}
		// Send two event control codes to the driver with an IOCTL_PMC_CONTROL call
		sprintf(szString, "Calling IOCTL_PMC_CONTROL with event codes:\nCode 1: %10x \nCode 2: %10x", evntsel[0], evntsel[1]);
		MessageBox(NULL, szString, "DoLoops",MB_OK);
		bReturnCode = DeviceIoControl(hDriver, IOCTL_PMC_CONTROL, (LPVOID)(evntsel), sizeof(evntsel), iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
		n = *iobuf;
		if (bReturnCode) {
		    sprintf(szString, "IOCTL_PMC_CONTROL succeeded;\nRETURNING %d bytes\nReturned value (nctrs): 0x%x", dwBytesReturned, n);
		    MessageBox(NULL, szString, "DoLoops",MB_OK);
		}
		else {
		    bReturnCode = GetLastError();
		    sprintf(szString, "IOCTL_PMC_CONTROL Failed\nLast Error Code: %x\nReturned value: %x", bReturnCode, n);
		    MessageBox(NULL, szString, "DoLoops", MB_OK);
		    CloseHandle(hDriver);
		    return;
		}
		for (i=0; i<n-1; i++)
			pmc[i] = read_pmc(i);	// read the counters
		do_flops(1000000);		// do some work
		for (i=0; i<n-1; i++)
			pmc[i] = read_pmc(i) - pmc[i];	// read the counters
		sprintf(szString, "I just completed 1,000,000 iterations of do_flops.\n");
		for (i=0; i<n-1; i++)
			sprintf(szString+strlen(szString),"PMC %d: %10lld \n", i, pmc[i]);
		MessageBox(NULL,szString,"DoLoops",MB_OK);
		CloseHandle(hDriver);
	}
}


// put the driver through its paces to make sure it's there and active
// this test has been modified to provide the content from the kern_info call
// only the names remain the same to confuse the innocent...
static void HelloTest(void)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    char szString[256]; // character buffer
    int iobuf[256];     // I/O buffer

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {
	// Send a request to the driver. The request code is HELLO, no parameters
	bReturnCode = DeviceIoControl(hDriver, HELLO, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
	if (bReturnCode) {
	  sprintf(szString, "PMC INFO RETURNED %d bytes: %s\n", dwBytesReturned, iobuf);
		MessageBox(NULL, szString, "PMC Info",MB_OK);
	}
	else 	MessageBox(NULL,"PMC Info failed.","PMC Info",MB_OK);
	CloseHandle(hDriver);
    }
}

// put the driver through its paces to make sure it's there and active
static void HelloNumTest(void)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    char szString[256]; // character buffer
    int iobuf[256];     // I/O buffer

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {
	// Send a request to the driver. The request code is HELLONUM, one integer parameter
	iobuf[0] = 319;    // my favorite number :-)
	bReturnCode = DeviceIoControl(hDriver, HELLONUM, iobuf, sizeof(int), iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
	if (bReturnCode) {
	  sprintf(szString, "HELLONUM RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
		MessageBox(NULL,szString, "HELLONUM Test",MB_OK);
	}
	else 	MessageBox(NULL,"HELLONUM failed.","HELLONUM Test",MB_OK);
	CloseHandle(hDriver);
    }
}


// put the driver through its paces to make sure it's there and active
static void TaskSwitchTest(void)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    char szString[256]; // character buffer
    int iobuf[256];     // I/O buffer

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {
	bReturnCode = DeviceIoControl(hDriver, TASKSWITCH, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
	if (bReturnCode) {
	    if (iobuf[0] == 0) {
		MessageBox(NULL, "This machine is running the Uniprocessor Free Build.\nTask Switching is not enabled.","TASKSWITCH Test",MB_OK);
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


LRESULT CALLBACK getFileHook(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	if (message == WM_INITDIALOG) centerDialog(GetParent(hDlg));
	return(FALSE);
}

//______________________________________________________________________________________
static short bottom, left;
static void centerDialog(HWND hdlg)
{
	RECT r;
	short top, screenWd, screenHt;

	// center a dialog on the screen
	GetWindowRect(hdlg, &r);
	screenWd = GetSystemMetrics(SM_CXSCREEN);
	screenHt = GetSystemMetrics(SM_CYSCREEN);
	left = (short)(screenWd - (r.right  - r.left)) / 2;
	top  = (short)(screenHt - (r.bottom - r.top)) / 3;
	bottom = top + (short)(r.bottom - r.top);
	SetWindowPos(hdlg, NULL, left, top, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
}

//______________________________________________________________________________________
static void centerDiagnostics(HWND hdlg)
{
	RECT r;

	// center a dialog on the screen
	GetWindowRect(hdlg, &r);
	SetWindowPos(hdlg, NULL, left, bottom - (r.bottom - r.top), 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
}

//______________________________________________________________________________________
// Support routines for the DoLoops button

// as found in papi/src/ctests/do_loops.c
volatile double a = 0.5, b = 2.2;
static void do_flops(int n)
{
   int i;
   double c = 0.11;

   for (i = 0; i < n; i++) {
      c += a * b;
   }
}

static unsigned __int64 read_pmc(int reg) {
    unsigned int v1, v2;
    __asm {
	mov ecx, reg
	rdpmc
	mov v1, eax
	and edx, 0x000000FF
	mov v2, edx
    };
    return(v1 + ((unsigned __int64)v2<<32));
}

