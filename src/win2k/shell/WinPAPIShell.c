// WinPAPIShell.c : Defines the entry point for the application.
//
#include "stdafx.h"
#include <winioctl.h>
#include <stdio.h>
#include "resource.h"
#include "winpmc.h"

#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;								// current instance
TCHAR szTitle[MAX_LOADSTRING];					// The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];			// The title bar text
TCHAR szLoadString[MAX_LOADSTRING];				// prompt text

// Foward declarations of functions included in this code module:
ATOM				MyRegisterClass(HINSTANCE hInstance);
BOOL				InitInstance(HINSTANCE, int);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK	About(HWND, UINT, WPARAM, LPARAM);


int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow)
{
 	// TODO: Place code here.
	MSG msg;
	HACCEL hAccelTable;

	// Initialize global strings
	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadString(hInstance, IDC_WINPAPISHELL, szWindowClass, MAX_LOADSTRING);
	LoadString(hInstance, IDS_HELLO, szLoadString, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	// Perform application initialization:
	if (!InitInstance (hInstance, nCmdShow)) 
	{
		return FALSE;
	}

	hAccelTable = LoadAccelerators(hInstance, (LPCTSTR)IDC_WINPAPISHELL);

	// Main message loop:
	while (GetMessage(&msg, NULL, 0, 0)) 
	{
		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) 
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	return msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
//  COMMENTS:
//
//    This function and its usage is only necessary if you want this code
//    to be compatible with Win32 systems prior to the 'RegisterClassEx'
//    function that was added to Windows 95. It is important to call this function
//    so that the application will get 'well formed' small icons associated
//    with it.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEX wcex;

	wcex.cbSize = sizeof(WNDCLASSEX); 

	wcex.style			= CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= (WNDPROC)WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= LoadIcon(hInstance, (LPCTSTR)IDI_WINPAPISHELL);
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName	= (LPCSTR)IDC_WINPAPISHELL;
	wcex.lpszClassName	= szWindowClass;
	wcex.hIconSm		= LoadIcon(wcex.hInstance, (LPCTSTR)IDI_SMALL);

	return RegisterClassEx(&wcex);
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
   HWND hWnd;

   hInst = hInstance; // Store instance handle in our global variable

   hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  FUNCTION: WndProc(HWND, unsigned, WORD, LONG)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND	- process the application menu
//  WM_PAINT	- Paint the main window
//  WM_DESTROY	- post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	RECT rt;
	HDC hdc;

	switch (message) 
	{
		case WM_COMMAND:
			wmId    = LOWORD(wParam); 
			wmEvent = HIWORD(wParam); 
			// Parse the menu selections:
			switch (wmId)
			{
                case IDM_TEST_KERNEL:
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
						MessageBox(hWnd,"Bummer","Driver Load Failed.",MB_OK);
					else {

						// Dispatch the READ_TEST_STRING IOCTL to our NT or 95 driver.
						bReturnCode = DeviceIoControl(hDriver,
									  IOCTL_PMC_READ_TEST_STRING,
									  0, 0,
									  iobuf, sizeof(iobuf),
									  &dwBytesReturned,
									  NULL);

						// Display the results!
						MessageBox(hWnd,(const char *)iobuf,"Get Driver Test String",MB_OK);
//						__asm mov ecx, 0x00000000
//						__asm rdpmc
//						MessageBox(hWnd,"We just successfully executed RDPMC from user space!","RDPMC Test...",MB_OK);

						/* Send a request to the driver. The request code is HELLO, no parameters */

						bReturnCode = DeviceIoControl(hDriver, HELLO, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
						if (bReturnCode)
						 {
						  sprintf(szString, "HELLO RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
						  MessageBox(hWnd, szString, "HELLO Test",MB_OK);
						}
						else MessageBox(hWnd,"HELLO failed.","HELLO Test",MB_OK);


						/* Send a request to the driver. The request code is HELLONUM, one integer parameter */

						iobuf[0] = 319;    // my favorite number :-)
						bReturnCode = DeviceIoControl(hDriver, HELLONUM, iobuf, sizeof(int), iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
						if (bReturnCode)
						 {
						  sprintf(szString, "HELLONUM RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
						  MessageBox(hWnd,szString, "HELLONUM Test",MB_OK);
						}
						else MessageBox(hWnd,"HELLONUM failed.","HELLONUM Test",MB_OK);

						/* Send a request to the driver. The request code is TASKSWITCH, no parameters */

						bReturnCode = DeviceIoControl(hDriver, TASKSWITCH, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
						if (bReturnCode)
						 {
						  sprintf(szString, "TASKSWITCH RETURNED %d bytes, number of task switches is %d\n", dwBytesReturned, iobuf[0]);
						  MessageBox(hWnd, szString, "TASKSWITCH Test",MB_OK);
						}
						else MessageBox(hWnd,"TASKSWITCH failed.","TASKSWITCH Test",MB_OK);


						CloseHandle(hDriver);
					}
                    break;
                }

				case IDM_DGEMM:
					LoadString(hInst, IDS_DGEMM, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 30);
					dgemm_test();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;

				case IDM_HILEVEL:
					LoadString(hInst, IDS_HILEVEL, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 30);
					PAPI_test_hilevel();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_FLOPS:
					LoadString(hInst, IDS_FLOPS, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 30);
					PAPI_test_flops();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_GET_CYCLES:
					LoadString(hInst, IDS_GET_CYCLES, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 30);
					test_get_cycles();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_STRINGS:
					LoadString(hInst, IDS_STRINGS, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 100, 40);
					PAPI_StringsAndLabels();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_ERRORS:
					LoadString(hInst, IDS_ERRORS, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 25);
					PAPI_Errors();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_NILS:
					LoadString(hInst, IDS_NILS, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 40);
					Nils_System_Info();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_AVAIL:
					LoadString(hInst, IDS_AVAIL, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_avail();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_COST:
					LoadString(hInst, IDS_COST, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 80, 60);
					PAPI_cost();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_CLOCKRES:
					LoadString(hInst, IDS_CLOCKRES, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_clock_res();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_ZERO:
					LoadString(hInst, IDS_ZERO, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_zero();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_FIRST:
					LoadString(hInst, IDS_FIRST, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 140, 40);
					PAPI_test_first();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_SECOND:
					LoadString(hInst, IDS_SECOND, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_second();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_THIRD:
					LoadString(hInst, IDS_THIRD, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_third();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_FOURTH:
					LoadString(hInst, IDS_FOURTH, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_fourth();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_FIFTH:
					LoadString(hInst, IDS_FIFTH, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_fifth();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_NINETH:
					LoadString(hInst, IDS_NINETH, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_nineth();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_TENTH:
					LoadString(hInst, IDS_TENTH, szLoadString, MAX_LOADSTRING);
					enterConsole(szLoadString, 120, 40);
					PAPI_test_tenth();
					exitConsole();
					GetClientRect(hWnd, &rt);
					InvalidateRect(hWnd, &rt, TRUE);
					break;
				case IDM_ABOUT:
				   DialogBox(hInst, (LPCTSTR)IDD_ABOUTBOX, hWnd, (DLGPROC)About);
				   break;
				case IDM_EXIT:
				   DestroyWindow(hWnd);
				   break;
				default:
				   return DefWindowProc(hWnd, message, wParam, lParam);
			}
			break;
		case WM_PAINT:
			hdc = BeginPaint(hWnd, &ps);
			// TODO: Add any drawing code here...
			GetClientRect(hWnd, &rt);
			DrawText(hdc, szLoadString, strlen(szLoadString), &rt, DT_CENTER);
			EndPaint(hWnd, &ps);
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
   }
   return 0;
}

// Mesage handler for about box.
LRESULT CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		case WM_INITDIALOG:
				return TRUE;

		case WM_COMMAND:
			if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL) 
			{
				EndDialog(hDlg, LOWORD(wParam));
				return TRUE;
			}
			break;
	}
    return FALSE;
}
