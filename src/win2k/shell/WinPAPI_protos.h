// prototypes for WinPAPI routines

// tests
extern dgemm_test(void); 
extern int PAPI_test_hilevel(void); 
extern int PAPI_test_flops(void); 
extern int PAPI_test_zero(void); 
extern int PAPI_test_first(void); 
extern int PAPI_test_second(void); 
extern int PAPI_test_third(void); 
extern int PAPI_test_avail(void); 
extern int PAPI_cost(void); 
extern int PAPI_clock_res(void); 
extern int PAPI_test_fourth(void); 
extern int PAPI_test_fifth(void); 
extern int PAPI_test_nineth(void); 
extern int PAPI_test_tenth(void); 
extern int Nils_System_Info(void);
extern int PAPI_StringsAndLabels(void);
extern int PAPI_Errors(void);
extern int test_get_cycles(void);

// winpapi_console
extern void enterConsole(LPSTR title, short nChars, short nLines);
extern void exitConsole(void);
extern void waitConsole(void);
extern DWORD resizeConBufAndWindow(HANDLE hConsole, SHORT xSize, SHORT ySize);
