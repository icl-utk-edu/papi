/* Based primarily on iprobe code from Compaq.
   Mods for alpha-linux by Glenn Laguna, Sandia National Lab, 
   galagun@sandia.gov
*/

#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <stropts.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>

#include <signal.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

#include <ipr_api_ext_defs.h>
#include <ipr_base_masks.h>
#include <ipr_events.h>
#include <iprobe_struct.h>

#define ulong_t long
#define IPR_ALL_CPUS    (0xffffffff)
#define IPR_SUCCESS(v) ((v) & 1)

#define max_counter 8

#define PF6_LOW_FREQ 1048576
#define PF6_HIGH_FREQ 272
#define PF5_LOW_FREQ 65536;
#define PF5_HIGH_FREQ 256

#define HW_SUCCESS             0

#define HW_FAILURE            -1

static IPR_CTRMASK CounterMask;
IPR_FREQ *FrequencyArray;
IPR_EVENT *EventArray;
IPR_FREQ *WhichFrequency;
IPR_EVENT *WhichEvent;

unsigned long BIprB2M (unsigned int b);

#define BIprIsSet(m,b) (((m) & BIprB2M(b)) != 0)

int printctrmask(unsigned long ctrmask);

/*                                                                            */
/*----------------------------------------------------------------------------*/
/* local variables                                                            */
IPR_HMASK  TypeOfHistogram;          /* kind of histos to collect */
IPR_HISTELEM *HistogramArea[1];         /* histogram vector loc */
IPR_SIZE HistogramSize[1];         /* histogram vector size */
IPR_HMASK HistogramType[1];         /* types of histograms */
IPR_HISTELEM *HistogramBuffers;      /* vector of histogram bufs */
IPR_HISTELEM **HistogramByCPU;
IPR_HISTELEM **HistogramTempData;
IPR_HISTELEM **HistogramTotalByCPU;
IPR_HISTELEM **HistogramByCPUData;
IPR_HISTELEM **HistogramTotalByCPUData;

FLAG BufferFullFlag = 0;     /* flag for buffer full condition */
FLAG CntrlCFlag = 0;    /* flag for CTRL-C hit by user */
FLAG TimeOutFlag = 0;   /* flag for end time expiration */

uint32 TotalNumberOfCounters;
uint32 TotalNumberOfCPUs = 1;



/* Stuff for the signal handlers */
time_t completion_time;

volatile uint32 *pTimerFlag=NULL;
volatile uint32 *pBufferFullFlag=NULL;
volatile uint32 *InterruptFlag=NULL;

void (*OldAlarmHandler)();
void (*OldInterruptHandler)();
void (*OldUser1Handler)();
void (*OldChildHandler)();

static int *cpu2fd_mapping = NULL;

/*                                                                            */
/*============================================================================*/
/* Starts hardware counters.                                                  */


int
HW_driver_start(int *commands, int proc_type)

{
  long  event_select, int_frequency;
  IPR_STATUS status;
  IPR_MMASK ModeMask;
  int32     ModeFlag;
  IPR_CTRMASK RequestedCounterMask;
  IPR_COUNT ncounters;
  uint64 MuxValue;
  uint32 CurrentCounter;
  uint32 CurrentCPU;
  int i;

  int ISA_EV5 = 0,  ISA_EV6 = 0;
  IPR_HMASK  TypeOfHistogram;          /* kind of histos to collect */

#ifdef DEBUG
  printf("\tin HW_driver_start, commands are: 0x%lx 0x%lx 0x%lx\n",
	 commands[0], commands[1], commands[2]); 
#endif

/* determine processor type */
  if(proc_type == 8)
    {
      ISA_EV6 = 1;
      TotalNumberOfCounters = 2;
    }
  else
    {
      ISA_EV5 = 1;
      TotalNumberOfCounters = 3;
    }

/*******************************************************************************/
/*****************Select Which Counter to use***********************************/
/*******************************************************************************/
#ifdef DEBUG
  printf("\tIn HW_driver_start, max_counter = %d, TotalNumberOfCounters = %d\n",
	 max_counter, TotalNumberOfCounters);
#endif

  event_select = int_frequency = 0;
      
  /* For a 21264, there are 2 counters, for now we will use only one */
  FrequencyArray = (IPR_FREQ *) calloc(TotalNumberOfCounters, sizeof(IPR_FREQ));
  EventArray = (IPR_EVENT *) calloc(TotalNumberOfCounters, sizeof(IPR_EVENT));
  WhichFrequency = (IPR_FREQ *) calloc(TotalNumberOfCounters, sizeof(IPR_FREQ));
  WhichEvent = (IPR_EVENT *) calloc(TotalNumberOfCounters, sizeof(IPR_EVENT));

/***********************************************************************************************/
/*************************************I hate this kind of shit**********************************/
/***********************************************************************************************/
  HistogramBuffers =(IPR_HISTELEM *) malloc(sizeof(IPR_HISTELEM) * TotalNumberOfCounters);

  HistogramByCPU = (IPR_HISTELEM **) malloc(sizeof(IPR_HISTELEM *) * TotalNumberOfCPUs);

  HistogramTotalByCPU =(IPR_HISTELEM **) malloc(sizeof(IPR_HISTELEM *) * TotalNumberOfCPUs);

  HistogramByCPUData =(IPR_HISTELEM **) malloc(sizeof(IPR_HISTELEM *) * TotalNumberOfCPUs);

  for(CurrentCPU = 0; CurrentCPU < TotalNumberOfCPUs; CurrentCPU++)
  {
    HistogramByCPUData[CurrentCPU] =(IPR_HISTELEM *) malloc(sizeof(IPR_HISTELEM) * TotalNumberOfCounters);
  }

  HistogramTempData =(IPR_HISTELEM **) malloc(sizeof(IPR_HISTELEM *) * TotalNumberOfCPUs);

  for(CurrentCPU = 0; CurrentCPU < TotalNumberOfCPUs; CurrentCPU++)
  {
    HistogramTempData[CurrentCPU] =(IPR_HISTELEM *) malloc(sizeof(IPR_HISTELEM) * TotalNumberOfCounters);
  }

  for(CurrentCPU = 0; CurrentCPU < TotalNumberOfCPUs; CurrentCPU++)
  {
    for(CurrentCounter = 0; CurrentCounter < TotalNumberOfCounters; CurrentCounter++)
    {
      HistogramTempData[CurrentCPU][CurrentCounter] = 0;
    }
  }

  HistogramTotalByCPUData =(IPR_HISTELEM **) malloc(sizeof(IPR_HISTELEM *) * TotalNumberOfCPUs);

  for(CurrentCPU = 0; CurrentCPU < TotalNumberOfCPUs; CurrentCPU++)
  {
    HistogramTotalByCPUData[CurrentCPU] =(IPR_HISTELEM *) malloc(sizeof(IPR_HISTELEM) * TotalNumberOfCounters);
  }


 for(CurrentCPU = 0; CurrentCPU < TotalNumberOfCPUs; CurrentCPU++)
  {
    HistogramByCPU[CurrentCPU]     = &HistogramByCPUData[CurrentCPU][0];
    HistogramTotalByCPU[CurrentCPU] = &HistogramTotalByCPUData[CurrentCPU][0];
  }

/************************* extract selected events *****************************************/

 if (ISA_EV5) {
     FrequencyArray[0] = FrequencyArray[1] = FrequencyArray[2] = PF5_LOW_FREQ;
 } else {
     FrequencyArray[0] = FrequencyArray[1] = PF6_LOW_FREQ;
 }

 ncounters = 0;
 for (i = 0; i < 3; ++i) {
   if (commands[i] != 0)
     {
	 EventArray[ncounters] = commands[i];
	 FrequencyArray[ncounters] = modify_freqs(commands[i],FrequencyArray[i]);
	 ncounters++;
     }
 }
     
/*******************************************************************************/
/*******************************Allocate Counters*******************************/
/*******************************************************************************/

  status = IprAllocate();

  if(!IPR_SUCCESS(status))
    {
      printf("IprAllocate failed with status %d\n",status);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else
      printf("IprAllocate successful\n");
#endif

/*******************************************************************************/
/****************************Create Histogram Buffers***************************/
/*******************************************************************************/

  /* Initialize histogram mask to collect address histograms */
  TypeOfHistogram.ipr_hmask_l_bits = 0;     /* clear all bits */
  TypeOfHistogram.ipr_hmask_v_total = 1;    /* set total bit */

  status = IprCreateHistogramBuffers(IPR_ALL_CPUS,      /* CPU mask */
				     TypeOfHistogram,   /* kind of hist data */
				     0,      /* no beginning addr */
				     0,      /* no end addr */
				     0,      /* no granularity */
				     0);     /* unknown param */
  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprCreateHistogramBuffers", status, NULL);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else
      printf("IprCreateHistogramBuffers successful\n");
#endif

/*******************************************************************************/
/***************************Create Signal Handlers******************************/
/*******************************************************************************/

  /* Establish a handler for the timer signal */
  status = OSDeclareWakeUpHandler(&TimeOutFlag);
  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("OSDeclareWakeUpHandler", status, NULL);
      return HW_FAILURE;
    }

/* Establish a signal handler for user-requested abort */
  status = OSDeclareCntrlCHandler(&CntrlCFlag);
  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("OSDeclareCntrlCHandler", status, NULL);
      return HW_FAILURE;
    }

/* Establish a signal handler for the PFM interrupt */
  status = OSDeclarePFMHandler(&BufferFullFlag);
  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("OSDeclarePFMHandler", status, NULL);
      return HW_FAILURE;
    }

/*******************************************************************************/
/**********************************Set Modes************************************/
/*******************************************************************************/
  /* set contexts in which to count */
  ModeFlag = IPR_MMASK_M_USER;
  ModeMask.ipr_mmask_l_bits = 0;

#ifdef DEBUG
  for (i = 0; i < ncounters; ++i) 
    printf("\tEventArray[%d] = %d, Frequency Array[%d] = %d\n",
	   i,EventArray[i],i,FrequencyArray[i]);
  printf("ModeMask = %ld\n",ModeMask);
#endif

  status = IprSpecifyOperations(IPR_ALL_CPUS,       /* cpu mask */
				EventArray,         /* event array */
				FrequencyArray,     /* freq array */
				ncounters,          /* sz of both arrays */
				ModeMask,           /* mode selection */
				&MuxValue,          /* MUX return */
				&RequestedCounterMask);     /* rtn active ctrs */

  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprSpecifyOperations", status, NULL);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else
    {
      printf("IprSpecifyOperations successful\n");
      printf("MuxValue = %ld, RequestedCounterMask = %ld\n",MuxValue,RequestedCounterMask);
      printf("ModesToUse = %ld\n",ModeMask);
    }
#endif

/*******************************************************************************/
/****************I'm not sure what these are for********************************/
/*******************************************************************************/

  status = IprEventCounterMap(IPR_CURRENT_CPU,   /* CPU to check */
			      WhichEvent,  /* vector of counters */
			      TotalNumberOfCounters);    /* vector length */

  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprEventCounterMap", status, NULL);
      return HW_FAILURE;
    }


  status = IprFreqCounterMap(IPR_CURRENT_CPU,    /* CPU to check */
			     WhichFrequency, /* return vector of freqs */
			     TotalNumberOfCounters);    /* vector length */
  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprFreqCounterMap", status, NULL);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else
    for (i = 0; i < TotalNumberOfCounters; ++i) 
      printf("\tWhichEvent[%d] = 0x%x, WhichFrequency[%d] = %d\n",
	     i,WhichEvent[i],i,WhichFrequency[i]);
#endif


/*******************************************************************************/
/************************Set Active Counter Mask********************************/
/*******************************************************************************/

  status = IprSetActiveCounters(IPR_ALL_CPUS, RequestedCounterMask, &CounterMask);

  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprSetActiveCounters", status, NULL);
      return HW_FAILURE;
    }

#ifdef DEBUG
  (void) printf("\tRequested counter mask is: 0%o\n", RequestedCounterMask);
  (void) printf("\tAnticipated counter mask is: 0%o\n", CounterMask);
#endif

  /*  printf("CoutrMsk = ");
      printctrmask(RequestedCounterMask);*/


/*******************************************************************************/
/*********************Set Additonal Histogram Stuff*****************************/
/*******************************************************************************/
  HistogramArea[0] = &HistogramBuffers[0]; /* histo addr */
  HistogramSize[0] = TotalNumberOfCounters;
  HistogramType[0].ipr_hmask_l_bits = IPR_HMASK_M_TOTAL;

  for(CurrentCounter = 0; CurrentCounter < TotalNumberOfCounters; CurrentCounter++)
  {
      HistogramByCPU[0][CurrentCounter] = 0;  /* iterative totals */
      HistogramTotalByCPU[0][CurrentCounter] = 0;  /* cumulative totals */
  }

/*******************************************************************************/
/*********************************Clear Counters********************************/
/*******************************************************************************/
  status = IprClear(IPR_ALL_CPUS, 0x3);

  if (! IPR_SUCCESS (status))
    {
      IprPrintMessage("IprClear", status, NULL);
      return HW_FAILURE;
    }

/*******************************************************************************/
/******************************Start the Counters*******************************/
/*******************************************************************************/
  status = IprStart(IPR_ALL_CPUS, CounterMask); 

  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprStart", status, NULL);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else 
      printf("\tCounter successfully started\n\n"); 
#endif

/*******************************************************************************/
/*******************************************************************************/

  /* get actual processor cycle counter value (to measure elapsed cycles) */
  /*  hw_descr->cycle_counter_value = read_cycle_counter();*/
  
  /* everything ok */
  return HW_SUCCESS;
}




/*                                                                            */
/*============================================================================*/
/* Clear all counters                                                     */


int
HW_driver_clear(void)
{
  IPR_STATUS status;

  status = IprClear(IPR_ALL_CPUS, 0x3);

  if (! IPR_SUCCESS (status))
    {
      IprPrintMessage("IprClear", status, NULL);
      return HW_FAILURE;
    }

  return HW_SUCCESS;
}





/*                                                                            */
/*============================================================================*/
/* Read hardware counters                                                     */

int
HW_driver_read(  long *counter_values,
		 int proc_type
		 )

{
  long cycles, read_cycles;
  unsigned long  used_mask;
  IPR_STATUS status;
  int CurrentCounter;
  int CPUIndex = 0;
  int CPUContext;
  int TotalNumberOfCounters;

  if(proc_type == 8)
      TotalNumberOfCounters = 2;
  else 
      TotalNumberOfCounters = 3;

  /* read out performance counters through driver interface */

  for(CPUContext = -1; CPUContext < IPR_MAX_CPUS;) 
    {
      status = IprReadMultipleHistograms(
				       IPR_ALL_CPUS,   /* eligible CPU mask */
				       &CPUContext,    /* cpu ctx buffer */
				       HistogramArea, /* ptr to hist bufs */
				       HistogramSize, /* sizes of bufs */
				       0,      /* min value array */
				       HistogramType, /* type array */
				       0,      /* unknown arg */
				       IPR_HACTION_COPY, /* collect action */
				       1);         /* array size */

      if(! IPR_SUCCESS(status))
	{
	  printf("\tIprReadMultipleHistograms failed\n");
	  exit(-1);
	}
    }

  if(!IPR_SUCCESS(status))
    {
      IprPrintMessage("IprReadAll", status, NULL);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else
      printf("Counters successfully read\n");
#endif

  for(CurrentCounter = 0; CurrentCounter < TotalNumberOfCounters; CurrentCounter++)
    {
      HistogramByCPU[CPUIndex][CurrentCounter] = HistogramBuffers[CurrentCounter];
      counter_values[CurrentCounter] = 
	HistogramBuffers[CurrentCounter] * FrequencyArray[CurrentCounter];
/* Apparantly we need this for TAU to give the right values */
      HistogramBuffers[CurrentCounter] = 0;

#ifdef DEBUG
      printf("HW-Count[%d] : %ld\n", CurrentCounter,counter_values[CurrentCounter]);
#endif
    }

  /* everything ok */
  return HW_SUCCESS;
}

/*============================================================================*
 *                             that's all folks                               *
 *============================================================================*/

FLAG *GetStartedProcessExited(void)
{
  static FLAG StartedProcessExited = 0;

  return &StartedProcessExited;
}


static void SignalHandler(int SignalNumber)
{

  FLAG *StartedProcessExited;
  time_t PresentTime;
  
  printf("*************Received Signal Number = %d\n",SignalNumber);

  StartedProcessExited = GetStartedProcessExited();

  /* Increment the appropriate flag, if there is an address */
  switch (SignalNumber)
    {
    case SIGUSR1: 
      if (pBufferFullFlag!=NULL)
	{	
	  (*pBufferFullFlag)++; 
	}     
      break;
    case SIGINT:  
      if (InterruptFlag!=NULL)
	{
	  (*InterruptFlag)++;
	}
      break;
    case SIGALRM:
      {
	/* THis is need because VMS is stupid, and it doesn't remeber which signal goes to
	   which handler. */

	signal(SIGALRM, SignalHandler);
	
	/*
	 *  Only set the timeout flag if this ALRM signal
	 *  occurred and the completion time is met or exceeded.
	 *  This allows both a scheduled wait time and an delta interval
	 *  time to work.
	 */
	if (pTimerFlag!=NULL)
	  {
	    PresentTime = time(0);
	    if (PresentTime >= completion_time)
	      {
		(*pTimerFlag)++;
	      }
	  }
	break;
      }
    case SIGCHLD:
     
      (*StartedProcessExited) = 1;
 
      /* This uninstalls the child death signal handler. */
      OldChildHandler = signal(SIGCHLD, OldChildHandler);

    default:
      break;
    }
}/* End SignalHandler() */


IPR_STATUS OSDeclareWakeUpHandler(FLAG *pTempTimerFlag)
{
volatile uint32 *pTimerFlag=NULL;
    if (pTempTimerFlag)
      {
	pTimerFlag = pTempTimerFlag;
      }

    OldAlarmHandler = signal(SIGALRM, SignalHandler);
    return 1;
}

IPR_STATUS OSDeclareCntrlCHandler(FLAG *pInterruptFlag)
{
    if (pInterruptFlag)
      {
	InterruptFlag = pInterruptFlag;
      }
    OldInterruptHandler = signal(SIGINT, SignalHandler);
    return 1;
}


IPR_STATUS OSDeclarePFMHandler(FLAG *pTempBufferFullFlag)
{
    if (pTempBufferFullFlag)
      {
	pBufferFullFlag = pTempBufferFullFlag;
      }
    OldUser1Handler = signal(SIGUSR1, SignalHandler);
    return 1;
}


void OSWaitForSignal(void)
{
    /* wait for a handled signal */
    (void) sigpause(sigmask(0));		/* always returns -1 */

}/* end OSWaitForSignal() */


IPR_STATUS OSScheduleRepeatingWakeUp(time_t DeltaTime)
{
    IPR_STATUS Status;		/* general status return */
    
    struct itimerval NewExpiration;

    /* if delta is negative, make it positive */
    if (DeltaTime < 0)
      {
	DeltaTime = -DeltaTime;
      }

    /* init interval times to 0 */
    memset(&NewExpiration, 0, sizeof (struct itimerval));

    /* Set the timer and the interval */
    NewExpiration.it_value.tv_sec = DeltaTime;
    NewExpiration.it_interval.tv_sec = DeltaTime;

    /* Establish the interval for a repeating interrupt */
    Status = setitimer(ITIMER_REAL, &NewExpiration, NULL);
    if (Status < 0)
      {
	perror("schedule_repeating_WakeUp: failure to set interval timer");
	return 0;
      }    

    return 1;
}/* end OSScheduleRepeatingWakeUp() */



int printctrmask(unsigned long ctrmask)
{
  int i;
  for (i = 63; i >=0; --i) 
    {
      if (BIprIsSet(ctrmask, i))
	printf("1");
      else
	printf("0");
    }
  printf("\n");
  return 0;
}


int
HW_driver_close(void)

{
  IPR_STATUS status;

/* Delete the buffers, now that we are done with them */
//  status = IprDeleteBuffers(IPR_ALL_CPUS);
//  if(!IPR_SUCCESS(status)) 
//    {
//      printf("Could not delete buffers, erro %X\n", status);
//      return HW_FAILURE;
//    }

  free_buffers();

  /* Releae the counters for other folks */
  status = IprDeallocate();
  if(!IPR_SUCCESS(status))
    {
      printf("Could not delallocate counters, erro %X\n", status);
      return HW_FAILURE;
    }
#ifdef DEBUG
  else
    printf("Counters successfully closed\n");
#endif

  /* everything ok */
  return HW_SUCCESS;
}


int free_buffers ()
{
  int CurrentCPU;
  free(FrequencyArray);

  free(EventArray);

  free(WhichFrequency);

  free(WhichEvent);

  free(HistogramBuffers);

  free(HistogramByCPU);

  free(HistogramTotalByCPU);

  for(CurrentCPU = 0; CurrentCPU < TotalNumberOfCPUs; CurrentCPU++)
  {
    free(HistogramByCPUData[CurrentCPU]);
    free(HistogramTempData[CurrentCPU]);
  }

  free(HistogramByCPUData);

  free( HistogramTempData);
  
  return(0);
}

int modify_freqs(int command, IPR_FREQ freq)
{
  IPR_FREQ new_freq;

  if (command == IPR_EVT_SCACHE_MISS)
    new_freq = PF5_HIGH_FREQ;
  else
    new_freq = freq;

  return new_freq;
}
